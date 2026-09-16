# ConvNeXt_V2_on_CIFAR_10_Image_Classification

Source repository: https://github.com/6657debug/ConvNeXt_V2_on_CIFAR_10_Image_Classification

This folder contains an AI-generated, self-contained PyTorch implementation of a
compact ConvNeXt V2 classifier for CIFAR-10. The program downloads the original
dataset, verifies its MD5 checksum, creates a deterministic 45,000/5,000
train/validation split, trains the model, selects the best validation checkpoint,
and evaluates the test set once at the end.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Fast verification (no dataset download)

```powershell
python convnextv2_cifar10.py --smoke-test --device cpu
```

## Full experiment

```powershell
python convnextv2_cifar10.py --epochs 100 --batch-size 128 --device auto
```

Outputs are written to `runs/convnextv2_cifar10/`: the best checkpoint,
per-epoch history, final metrics, and a 10-by-10 confusion matrix.

## Measured result

The complete 100-epoch run with seed 3024 was executed on an NVIDIA GeForce
RTX 4060 Laptop GPU using PyTorch 2.12.0+cu130:

- Best validation accuracy: **83.96%** at epoch 88
- Test accuracy: **82.65%** (8,265 / 10,000)
- Test cross-entropy loss: **0.6158**
- Runtime: **2,806.3 seconds**
- Trainable parameters: **3,389,170**

The machine-readable results are included under `runs/convnextv2_cifar10/`.

For a lower-memory GPU, reduce `--batch-size` to 64 or 32. CPU training is valid
but slow. The test set is not used for model selection.
