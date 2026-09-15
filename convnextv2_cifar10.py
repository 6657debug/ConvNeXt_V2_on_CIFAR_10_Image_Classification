"""Train a CIFAR-10 classifier with a compact ConvNeXt V2 model.

This implementation is self-contained apart from PyTorch, NumPy, and Pillow.
It downloads the original CIFAR-10 Python archive from the University of
Toronto, verifies its MD5 checksum, and does not require torchvision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract while rejecting entries that escape the destination."""
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    try:
        archive.extractall(destination, filter="data")
    except TypeError:  # Python < 3.12
        archive.extractall(destination)


def prepare_cifar10(data_dir: Path) -> Path:
    extracted = data_dir / "cifar-10-batches-py"
    expected = extracted / "test_batch"
    if expected.exists():
        return extracted

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "cifar-10-python.tar.gz"
    if not archive_path.exists() or md5(archive_path) != CIFAR10_MD5:
        print(f"Downloading CIFAR-10 to {archive_path} ...")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)
    actual_md5 = md5(archive_path)
    if actual_md5 != CIFAR10_MD5:
        raise RuntimeError(
            f"CIFAR-10 checksum mismatch: expected {CIFAR10_MD5}, got {actual_md5}"
        )
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, data_dir)
    return extracted


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def load_cifar10(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = prepare_cifar10(data_dir)
    train_parts = [_read_batch(root / f"data_batch_{i}") for i in range(1, 6)]
    train_images = np.concatenate([part[0] for part in train_parts], axis=0)
    train_labels = np.concatenate([part[1] for part in train_parts], axis=0)
    test_images, test_labels = _read_batch(root / "test_batch")
    return train_images, train_labels, test_images, test_labels


class CIFAR10Dataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = Image.fromarray(self.images[index])
        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment:
            array = np.pad(array, ((4, 4), (4, 4), (0, 0)), mode="reflect")
            top = random.randint(0, 8)
            left = random.randint(0, 8)
            array = array[top : top + 32, left : left + 32]
            if random.random() < 0.5:
                array = array[:, ::-1]
        array = (array - CIFAR10_MEAN) / CIFAR10_STD
        array = np.ascontiguousarray(array.transpose(2, 0, 1))
        return torch.from_numpy(array), torch.tensor(self.labels[index], dtype=torch.long)


class DropPath(nn.Module):
    """Stochastic depth applied independently to each sample."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class GRN(nn.Module):
    """Global Response Normalization for channels-last tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        spatial_norm = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
        normalized = spatial_norm / (spatial_norm.mean(dim=-1, keepdim=True) + self.eps)
        return x + self.gamma * (x * normalized) + self.beta


class ConvNeXtV2Block(nn.Module):
    def __init__(self, channels: int, drop_path: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.grn = GRN(4 * channels)
        self.project = nn.Linear(4 * channels, channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.project(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXtV2CIFAR(nn.Module):
    """ConvNeXt V2-Atto widths/depths with a CIFAR-sized stem."""

    def __init__(
        self,
        num_classes: int = 10,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        dims: tuple[int, ...] = (40, 80, 160, 320),
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt V2 requires four stages")

        # A 2x2/2 stem preserves more spatial detail than the ImageNet 4x4/4 stem.
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=2, stride=2),
                          ChannelFirstLayerNorm(dims[0], eps=1e-6)),
            *[
                nn.Sequential(
                    ChannelFirstLayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                for i in range(3)
            ],
        ])

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        stages: list[nn.Sequential] = []
        for depth, dim in zip(depths, dims):
            blocks = [
                ConvNeXtV2Block(dim, rates[cursor + block_index])
                for block_index in range(depth)
            ]
            stages.append(nn.Sequential(*blocks))
            cursor += depth
        self.stages = nn.ModuleList(stages)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        return self.norm(x.mean(dim=(-2, -1)))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x))


class ChannelFirstLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)
        for target, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            confusion[target, prediction] += 1
    return total_loss / total_examples, total_correct / total_examples, confusion


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
    return total_loss / total_examples, total_correct / total_examples


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_images, train_labels, test_images, test_labels = load_cifar10(args.data_dir)
    generator = np.random.default_rng(args.seed)
    indices = generator.permutation(len(train_labels))
    val_indices, train_indices = indices[:5000], indices[5000:]

    train_set = CIFAR10Dataset(train_images[train_indices], train_labels[train_indices], True)
    val_set = CIFAR10Dataset(train_images[val_indices], train_labels[val_indices], False)
    test_set = CIFAR10Dataset(test_images, test_labels, False)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def smoke_test(device: torch.device) -> None:
    model = ConvNeXtV2CIFAR().to(device)
    images = torch.randn(2, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1], device=device)
    logits = model(images)
    if logits.shape != (2, 10) or not torch.isfinite(logits).all():
        raise RuntimeError(f"Invalid model output: shape={tuple(logits.shape)}")
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    gradient_count = sum(p.grad is not None for p in model.parameters())
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"Smoke test passed: output={tuple(logits.shape)}, "
        f"loss={loss.item():.4f}, parameters={parameter_count:,}, "
        f"tensors_with_grad={gradient_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/convnextv2_cifar10"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")
    if args.smoke_test:
        smoke_test(device)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders(args)
    model = ConvNeXtV2CIFAR().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -math.inf
    best_path = args.output_dir / "best_model.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy, _ = evaluate(model, val_loader, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "class_names": CLASS_NAMES,
                    "args": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
                },
                best_path,
            )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_accuracy, confusion = evaluate(model, test_loader, device)
    np.savetxt(args.output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    final = {
        "best_epoch": checkpoint["epoch"],
        "best_validation_accuracy": checkpoint["val_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "elapsed_seconds": time.time() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
