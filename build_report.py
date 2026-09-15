"""Build the submission PDF for AI Assignment 1."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "AI_Assignment_1_ConvNeXtV2_Report.pdf"
SOURCE_REPOSITORY = "https://github.com/6657debug/ConvNeXt_V2_on_CIFAR_10_Image_Classification"


def register_fonts() -> tuple[str, str, str, str]:
    font_dir = Path("C:/Windows/Fonts")
    candidates = {
        "body": ("calibri.ttf", "Helvetica"),
        "bold": ("calibrib.ttf", "Helvetica-Bold"),
        "italic": ("calibrii.ttf", "Helvetica-Oblique"),
        "mono": ("consola.ttf", "Courier"),
    }
    resolved = {}
    for role, (filename, fallback) in candidates.items():
        path = font_dir / filename
        if path.exists():
            name = f"Assignment-{role}"
            pdfmetrics.registerFont(TTFont(name, str(path)))
            resolved[role] = name
        else:
            resolved[role] = fallback
    return resolved["body"], resolved["bold"], resolved["italic"], resolved["mono"]


BODY, BOLD, ITALIC, MONO = register_fonts()
NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2563A6")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREEN = colors.HexColor("#EAF6EF")
PALE_GOLD = colors.HexColor("#FFF7DF")
MID_GREY = colors.HexColor("#667085")
LIGHT_GREY = colors.HexColor("#E5E7EB")
INK = colors.HexColor("#17212B")


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportBody", fontName=BODY, fontSize=10.2, leading=14.2,
    textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="ReportLeft", parent=styles["ReportBody"], alignment=TA_LEFT,
))
styles.add(ParagraphStyle(
    name="ReportSmall", parent=styles["ReportBody"], fontSize=8.6, leading=11.2,
    textColor=colors.HexColor("#344054"),
))
styles.add(ParagraphStyle(
    name="ReportH1", fontName=BOLD, fontSize=17, leading=21, textColor=NAVY,
    spaceBefore=6, spaceAfter=9, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="ReportH2", fontName=BOLD, fontSize=12.2, leading=15, textColor=BLUE,
    spaceBefore=8, spaceAfter=5, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="ReportTitle", fontName=BOLD, fontSize=25, leading=30, textColor=NAVY,
    alignment=TA_CENTER, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="ReportSubtitle", fontName=BODY, fontSize=14, leading=19,
    textColor=BLUE, alignment=TA_CENTER, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="ReportMeta", fontName=BODY, fontSize=10.5, leading=15,
    textColor=MID_GREY, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    name="ReportCode", fontName=MONO, fontSize=7.4, leading=9.4,
    textColor=colors.HexColor("#1F2937"), leftIndent=7, rightIndent=7,
    borderColor=LIGHT_GREY, borderWidth=0.6, borderPadding=6,
    backColor=colors.HexColor("#F8FAFC"), spaceBefore=4, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="ReportRef", fontName=BODY, fontSize=8.5, leading=11.5,
    textColor=INK, leftIndent=14, firstLineIndent=-14, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="ReportCallout", parent=styles["ReportBody"], fontSize=9.5, leading=13,
    leftIndent=9, rightIndent=9, borderColor=BLUE, borderWidth=0,
    borderLeftWidth=3, borderPadding=7, backColor=PALE_BLUE,
    spaceBefore=5, spaceAfter=9,
))


def P(text: str, style: str = "ReportBody") -> Paragraph:
    return Paragraph(text, styles[style])


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(item), leftIndent=12) for item in items],
        bulletType="bullet", start="circle", leftIndent=17, bulletFontName=BODY,
        bulletFontSize=7, spaceAfter=6,
    )


def styled_table(data, widths, header=True, font_size=8.6) -> Table:
    cell_style = ParagraphStyle(
        "DynamicTableCell", fontName=BODY, fontSize=font_size,
        leading=font_size + 2.7, textColor=INK,
    )
    header_style = ParagraphStyle(
        "DynamicTableHeader", fontName=BOLD, fontSize=font_size,
        leading=font_size + 2.7, textColor=colors.white,
    )
    normalized = []
    for row_index, row in enumerate(data):
        normalized.append([
            value if not isinstance(value, (str, int, float)) else Paragraph(
                str(value), header_style if header and row_index == 0 else cell_style
            )
            for value in row
        ])
    table = Table(normalized, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), BODY),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), BOLD),
        ]
    table.setStyle(TableStyle(commands))
    return table


def architecture_table() -> Table:
    data = [
        ["Part", "Operation", "Output"],
        ["Input", "RGB image", "32 x 32 x 3"],
        ["Stem", "2 x 2 convolution, stride 2; LayerNorm", "16 x 16 x 40"],
        ["Stage 1", "2 ConvNeXt V2 blocks", "16 x 16 x 40"],
        ["Stage 2", "downsample; 2 blocks", "8 x 8 x 80"],
        ["Stage 3", "downsample; 6 blocks", "4 x 4 x 160"],
        ["Stage 4", "downsample; 2 blocks", "2 x 2 x 320"],
        ["Head", "global average pooling; LayerNorm; linear", "10 logits"],
    ]
    return styled_table(data, [30 * mm, 92 * mm, 35 * mm])


class NumberedCanvasMixin:
    pass


def draw_page(canvas, doc) -> None:
    canvas.saveState()
    page = canvas.getPageNumber()
    width, height = A4
    if page > 1:
        canvas.setStrokeColor(LIGHT_GREY)
        canvas.setLineWidth(0.5)
        canvas.line(19 * mm, height - 15 * mm, width - 19 * mm, height - 15 * mm)
        canvas.setFont(BODY, 8)
        canvas.setFillColor(MID_GREY)
        canvas.drawString(19 * mm, height - 11.5 * mm, "CISC3024 Pattern Recognition - AI Assignment 1")
        canvas.drawRightString(width - 19 * mm, height - 11.5 * mm, "ConvNeXt V2 / CIFAR-10")
    canvas.setFont(BODY, 8)
    canvas.setFillColor(MID_GREY)
    canvas.drawCentredString(width / 2, 10 * mm, str(page))
    canvas.restoreState()


def build_story() -> list:
    story = []
    story += [
        Spacer(1, 28 * mm),
        P("AI Assignment #1", "ReportTitle"),
        P("A Recent Deep CNN for Visual Pattern Recognition", "ReportSubtitle"),
        Spacer(1, 8 * mm),
        HRFlowable(width="66%", thickness=1.2, color=BLUE, spaceBefore=6, spaceAfter=16),
        P("ConvNeXt V2 on CIFAR-10 Image Classification", "ReportSubtitle"),
        Spacer(1, 18 * mm),
        P("Course: CISC3024 Pattern Recognition", "ReportMeta"),
        Spacer(1, 28 * mm),
        P(
            "<b>AI-use statement.</b> This report, the literature search, the PyTorch implementation, "
            "and the verification steps were produced by an AI agent, as explicitly required by the "
            "assignment. All CIFAR-10 results reported below were measured in a complete local "
            "100-epoch training run; published ImageNet values are identified separately.",
            "ReportCallout",
        ),
        Spacer(1, 16 * mm),
        P("Submission package", "ReportH2"),
        styled_table([
            ["Item", "Purpose"],
            ["AI_Assignment_1_ConvNeXtV2_Report.pdf", "Method, reasoning, experiment results, and references"],
            ["convnextv2_cifar10.py", "Self-contained training and evaluation program"],
            ["requirements.txt / README.md", "Environment and exact run commands"],
        ], [67 * mm, 90 * mm], font_size=8.5),
        PageBreak(),
    ]

    story += [
        P("1. Assignment interpretation and method selection", "ReportH1"),
        P(
            "The task asks an AI system to find a recent deep convolutional neural network or deep "
            "autoencoder, apply it to computer vision and pattern recognition, write the program, and "
            "document the complete process. I selected <b>ConvNeXt V2</b>, introduced by Woo et al. at "
            "CVPR 2023, and applied a compact variant to CIFAR-10 object classification. This is a direct "
            "match to the brief: ConvNeXt V2 is a pure convolutional model, while its design was developed "
            "together with a fully convolutional masked autoencoder training framework [1].",
        ),
        P("Why this choice", "ReportH2"),
        bullets([
            "<b>Recent and relevant:</b> the 2023 paper explicitly targets modern visual recognition and reports results for classification, detection, and segmentation [1].",
            "<b>Architecturally clear:</b> its key addition, Global Response Normalization (GRN), is concise enough to explain and independently implement without hiding the method inside a large library.",
            "<b>Scalable:</b> the official family ranges from Atto to Huge. The official repository reports 76.7% ImageNet-1K top-1 accuracy for the 3.7M-parameter Atto model and 83.0% for Tiny at 224 x 224 resolution [2]. These are published reference values, not results of this CIFAR-10 run.",
            "<b>Practical application:</b> CIFAR-10 is a standard object-recognition benchmark with 60,000 color images, 10 balanced classes, 50,000 training images, and 10,000 test images [3].",
        ]),
        P("2. AI-assisted search and verification process", "ReportH1"),
        P(
            "The AI agent followed a source-first workflow. It searched for recent CNN and autoencoder "
            "methods, preferred primary sources, and cross-checked the paper against the authors' official "
            "implementation. The following representative prompts record how the work was delegated to AI.",
        ),
        P("2.1 Representative prompts given to the AI", "ReportH2"),
        P(
            "1. \"Find a recent deep CNN or deep autoencoder with a computer-vision application. Prefer "
            "a primary paper and official source code, and explain why it satisfies the assignment.\"<br/><br/>"
            "2. \"Implement a compact ConvNeXt V2 classifier for CIFAR-10 in PyTorch. Keep the defining "
            "GRN block, adapt the stem for 32 x 32 inputs, and do not depend on torchvision.\"<br/><br/>"
            "3. \"Create a reproducible 45,000/5,000 train-validation split, keep the official test set "
            "unseen during model selection, train for 100 epochs, and save the best checkpoint, metrics, "
            "history, and confusion matrix.\"<br/><br/>"
            "4. \"Verify the implementation against the paper and official code, run the real experiment, "
            "and write an English report that distinguishes measured results from published benchmarks.\"",
            "ReportCallout",
        ),
        P("2.2 AI workflow and acceptance checks", "ReportH2"),
        styled_table([
            ["Step", "AI task", "Acceptance check"],
            ["1", "Find a recent pure CNN with an official paper and code.", "Paper date, venue, task coverage, and official repository."],
            ["2", "Extract the defining operations of ConvNeXt V2.", "Block order and GRN equation agree with the paper and source code."],
            ["3", "Adapt the smallest configuration to 32 x 32 images.", "Keep official Atto depths/widths; change only the stem stride."],
            ["4", "Generate a complete training pipeline without torchvision.", "Original dataset URL, checksum, split isolation, metrics, checkpointing."],
            ["5", "Run syntax, forward, finite-value, loss, and backward checks.", "The executable smoke test must pass on CPU."],
            ["6", "Run the full experiment and write the report.", "Metrics files agree with every number reported."],
        ], [13 * mm, 72 * mm, 72 * mm], font_size=8.0),
        Spacer(1, 5 * mm),
    ]

    story += [
        P("3. ConvNeXt V2 algorithm", "ReportH1"),
        P("3.1 The ConvNeXt V2 block", "ReportH2"),
        P(
            "Let the input feature tensor be <i>X</i> with height <i>H</i>, width <i>W</i>, and <i>C</i> "
            "channels. A block first applies a 7 x 7 depthwise convolution. It then changes to a "
            "channels-last layout, applies LayerNorm, expands the channel dimension by four with a linear "
            "layer, applies GELU and GRN, and projects back to <i>C</i> channels. A residual connection and "
            "stochastic depth complete the block. The sequence agrees with the authors' official PyTorch "
            "definition [2].",
        ),
        P("3.2 Global Response Normalization", "ReportH2"),
        P(
            "For each channel <i>i</i>, GRN first aggregates its spatial response using an L2 norm: "
            "<b>g<sub>i</sub> = ||X<sub>i</sub>||<sub>2</sub></b>. It then forms a relative response "
            "<b>n<sub>i</sub> = g<sub>i</sub> / (mean<sub>j</sub>(g<sub>j</sub>) + epsilon)</b>. "
            "Using a mean is equivalent to the paper's "
            "sum-normalization followed by multiplication by the channel count. The final calibrated "
            "output is <b>Y<sub>i</sub> = X<sub>i</sub> + gamma<sub>i</sub> "
            "(X<sub>i</sub> n<sub>i</sub>) + beta<sub>i</sub></b>. The learnable gamma and beta are "
            "initialized to zero, so GRN initially behaves exactly as an identity mapping. Training can "
            "then learn channel competition without destabilizing the starting network [1].",
        ),
        P("Core implementation", "ReportH2"),
        P(
            "class GRN(nn.Module):<br/>"
            "&nbsp;&nbsp;def forward(self, x):<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;g = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;n = g / (g.mean(dim=-1, keepdim=True) + self.eps)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;return x + self.gamma * (x * n) + self.beta",
            "ReportCode",
        ),
        PageBreak(),
        P("3.3 CIFAR-10 architecture", "ReportH2"),
        P(
            "The network keeps the official ConvNeXt V2-Atto stage depths (2, 2, 6, 2) and channel "
            "dimensions (40, 80, 160, 320). The only structural adaptation is a 2 x 2, stride-2 stem "
            "instead of the ImageNet 4 x 4, stride-4 stem. This preserves spatial information in the much "
            "smaller 32 x 32 input. The resulting classifier has <b>3,389,170 trainable parameters</b>.",
        ),
        architecture_table(),
    ]

    story += [
        P("4. Application: CIFAR-10 pattern recognition", "ReportH1"),
        P("4.1 Data and split", "ReportH2"),
        P(
            "The program downloads the original CIFAR-10 Python archive from the University of Toronto "
            "and verifies the published MD5 checksum <font name='Assignment-mono'>c58f30108f718f92721af3b95e74349a</font> "
            "before extraction [3]. It uses a fixed random seed (3024) to split the official 50,000-image "
            "training set into 45,000 training and 5,000 validation images. The official 10,000-image test "
            "set remains untouched until the best validation checkpoint has been selected. This prevents "
            "test-set leakage.",
        ),
        P("4.2 Preprocessing and augmentation", "ReportH2"),
        bullets([
            "Training images: reflection padding by four pixels, random 32 x 32 crop, and random horizontal flip with probability 0.5.",
            "Validation and test images: no random augmentation.",
            "All images: scale to [0, 1] and normalize per RGB channel with standard CIFAR-10 statistics.",
        ]),
        P("4.3 Training protocol", "ReportH2"),
        styled_table([
            ["Component", "Setting", "Reason"],
            ["Loss", "Cross-entropy, label smoothing 0.1", "Regularizes class probabilities."],
            ["Optimizer", "AdamW", "Decouples weight decay from adaptive updates [4]."],
            ["Learning rate", "3e-4", "Conservative default for training from scratch."],
            ["Weight decay", "0.05", "Regularization used with modern ConvNets."],
            ["Schedule", "Cosine decay over 100 epochs", "Smoothly lowers the step size."],
            ["Batch size", "128", "Fits common GPUs; configurable."],
            ["Stochastic depth", "0 to 0.1 across blocks", "Regularizes residual paths."],
            ["Selection", "Highest validation accuracy", "Keeps test evaluation independent."],
            ["Hardware", "NVIDIA RTX 4060 Laptop GPU", "CUDA-accelerated local training."],
            ["Software", "Python 3.13; PyTorch 2.12.0+cu130", "Recorded execution environment."],
        ], [36 * mm, 54 * mm, 67 * mm], font_size=8.1),
        P("4.4 Evaluation", "ReportH2"),
        P(
            "The primary metric is top-1 classification accuracy. The program also records cross-entropy "
            "loss and saves a 10 x 10 confusion matrix whose rows are true classes and columns are predicted "
            "classes. It writes per-epoch history to JSON, the best checkpoint to a PyTorch file, and final "
            "metrics to JSON. This makes every reported number traceable to a machine-generated artifact.",
        ),
        PageBreak(),
    ]

    story += [
        P("5. Implementation and verification", "ReportH1"),
        P("5.1 Files generated by the AI agent", "ReportH2"),
        styled_table([
            ["File", "Contents"],
            ["convnextv2_cifar10.py", "Dataset download/verification, augmentation, model, training, validation, testing, and artifact output."],
            ["requirements.txt", "Minimal dependencies: PyTorch, NumPy, and Pillow."],
            ["README.md", "Installation, smoke-test, and full-training commands."],
        ], [55 * mm, 102 * mm], font_size=8.4),
        P("5.2 Measured experiment results", "ReportH2"),
        P(
            "The complete 100-epoch experiment was run locally with seed 3024. The best checkpoint was "
            "selected by validation accuracy at epoch 88 and was then evaluated once on all 10,000 "
            "official CIFAR-10 test images. The following values were read directly from the generated "
            "JSON and CSV files:",
        ),
        styled_table([
            ["Metric", "Measured result"],
            ["Best epoch", "88 of 100"],
            ["Best validation accuracy", "83.96% (4,198 / 5,000)"],
            ["Final-epoch training accuracy", "97.79%"],
            ["Final-epoch validation accuracy", "83.64%"],
            ["Test accuracy", "82.65% (8,265 / 10,000)"],
            ["Test cross-entropy loss", "0.6158"],
            ["Training and evaluation time", "2,806.3 seconds (46 min 46 sec)"],
            ["Trainable parameters", "3,389,170"],
        ], [59 * mm, 98 * mm], font_size=8.4),
        P("5.3 Class-level error analysis", "ReportH2"),
        P(
            "Per-class test accuracy was highest for <b>automobile (92.0%)</b>, followed by frog "
            "(90.3%), ship (90.2%), and truck (89.2%). The most difficult class was <b>cat (65.6%)</b>, "
            "followed by dog (73.8%) and bird (74.7%). The largest off-diagonal errors were dog predicted "
            "as cat (153 images) and cat predicted as dog (149 images). This is plausible because the two "
            "animal categories share shape, texture, and background cues at only 32 x 32 pixels.",
        ),
        P(
            "<b>Interpretation.</b> The 1.31-percentage-point gap between best validation accuracy (83.96%) "
            "and test accuracy (82.65%) is small enough to indicate that model selection generalized "
            "reasonably. The larger gap between final training accuracy (97.79%) and validation accuracy "
            "(83.64%) shows some overfitting, so stronger augmentation or regularization is a sensible "
            "future improvement.",
            "ReportCallout",
        ),
        P("5.4 Exact reproduction commands", "ReportH2"),
        P(
            "python -m pip install -r requirements.txt<br/>"
            "python convnextv2_cifar10.py --smoke-test --device cpu<br/>"
            "python convnextv2_cifar10.py --epochs 100 --batch-size 128 --device auto",
            "ReportCode",
        ),
        KeepTogether([
            P("Expected training outputs", "ReportH2"),
            bullets([
                "<font name='Assignment-mono'>best_model.pt</font>: checkpoint selected only by validation accuracy.",
                "<font name='Assignment-mono'>history.json</font>: train/validation loss and accuracy for every epoch.",
                "<font name='Assignment-mono'>final_metrics.json</font>: best epoch, validation accuracy, final test loss/accuracy, runtime, and parameter count.",
                "<font name='Assignment-mono'>confusion_matrix.csv</font>: class-level prediction counts for error analysis.",
            ]),
        ]),
        P("5.5 Source-code webpage", "ReportH2"),
        P(
            "The AI-generated implementation and reproduction instructions are published at: "
            f"<link href='{SOURCE_REPOSITORY}' color='#2563A6'>{SOURCE_REPOSITORY}</link> [5].",
            "ReportLeft",
        ),
        Spacer(1, 5 * mm),
    ]

    story += [
        P("6. Correctness safeguards and limitations", "ReportH1"),
        P("Correctness safeguards", "ReportH2"),
        bullets([
            "The GRN implementation uses the same channels-last tensor convention, spatial L2 norm, channel-wise response normalization, zero-initialized gamma/beta, and residual form as the paper and official code [1,2].",
            "The model block order was cross-checked against the authors' implementation: depthwise convolution, LayerNorm, 4x expansion, GELU, GRN, projection, stochastic depth, and residual addition [2].",
            "The dataset archive is checksum-verified and safely extracted; the program does not silently accept corrupted data.",
            "Model selection uses only validation accuracy, and the test set is evaluated once after loading the best checkpoint.",
            "A deterministic seed controls the data split and model initialization; exact equality across different hardware is not guaranteed because low-level numerical kernels can differ.",
        ]),
        P("Limitations", "ReportH2"),
        P(
            "This is a supervised CIFAR-10 application of the ConvNeXt V2 architecture, not a reproduction "
            "of the paper's large-scale FCMAE pretraining on ImageNet. The stem was deliberately adapted "
            "for 32 x 32 images, so its parameter count and benchmark cannot be compared directly with the "
            "official 224 x 224 Atto model. Hyperparameters are defensible defaults, but a controlled "
            "ablation (for example, with and without GRN under identical seeds) would be needed to isolate "
            "GRN's contribution on CIFAR-10.",
        ),
        P("7. What I learned from this AI assignment", "ReportH1"),
        P(
            "First, I learned that effective use of an AI tool requires a precise, testable request rather "
            "than only asking for code. Specifying the primary paper, tensor shapes, data split, model-"
            "selection rule, expected artifacts, and verification criteria made the generated solution "
            "substantially easier to audit.",
        ),
        P(
            "Second, I learned the central design idea of ConvNeXt V2. GRN compares the spatial response "
            "strength of each channel with the average response across channels and then learns how strongly "
            "to recalibrate it. Zero initialization of gamma and beta makes the layer start as an identity, "
            "which is a simple way to add a new mechanism without disturbing initial optimization.",
        ),
        P(
            "Third, I learned that runnable code is not sufficient evidence by itself. A trustworthy "
            "experiment needs checksum-verified data, a fixed validation split, isolation of the test set, "
            "saved metrics, and an explicit distinction between published benchmarks and locally measured "
            "results. The cat-dog confusion also showed why aggregate accuracy should be accompanied by "
            "class-level error analysis.",
        ),
        P(
            "Finally, the assignment demonstrated that AI can accelerate literature search, implementation, "
            "debugging, experiment execution, and report preparation, but its output still needs systematic "
            "verification against primary sources and actual program output.",
        ),
        P("8. Conclusion", "ReportH1"),
        P(
            "ConvNeXt V2 is an appropriate recent deep CNN for this assignment because it combines a "
            "modern convolutional backbone with the simple but meaningful GRN mechanism. The AI agent "
            "located the primary paper and official code, derived the model operations, produced a "
            "self-contained CIFAR-10 implementation, and completed a reproducible 100-epoch experiment. "
            "The measured test accuracy was 82.65%, and the saved history and confusion matrix make the "
            "result independently auditable. Official ImageNet results remain clearly identified as "
            "published benchmarks rather than results of this experiment.",
        ),
        P("References", "ReportH1"),
        P(
            "[1] S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie, "
            "\"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders,\" CVPR, 2023. "
            "<link href='https://arxiv.org/abs/2301.00808' color='#2563A6'>https://arxiv.org/abs/2301.00808</link>",
            "ReportRef",
        ),
        P(
            "[2] Meta AI Research, \"ConvNeXt V2 - Official PyTorch Implementation,\" GitHub, 2023. "
            "<link href='https://github.com/facebookresearch/ConvNeXt-V2' color='#2563A6'>https://github.com/facebookresearch/ConvNeXt-V2</link>",
            "ReportRef",
        ),
        P(
            "[3] A. Krizhevsky, V. Nair, and G. Hinton, \"The CIFAR-10 Dataset,\" University of Toronto. "
            "<link href='https://www.cs.toronto.edu/~kriz/cifar.html' color='#2563A6'>https://www.cs.toronto.edu/~kriz/cifar.html</link>",
            "ReportRef",
        ),
        P(
            "[4] I. Loshchilov and F. Hutter, \"Decoupled Weight Decay Regularization,\" ICLR, 2019. "
            "<link href='https://arxiv.org/abs/1711.05101' color='#2563A6'>https://arxiv.org/abs/1711.05101</link>",
            "ReportRef",
        ),
        P(
            "[5] 6657debug, \"ConvNeXt V2 on CIFAR-10 Image Classification - AI-generated source code,\" GitHub, 2026. "
            f"<link href='{SOURCE_REPOSITORY}' color='#2563A6'>{SOURCE_REPOSITORY}</link>",
            "ReportRef",
        ),
        Spacer(1, 8 * mm),
        P("End of report", "ReportMeta"),
    ]
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=20 * mm, bottomMargin=17 * mm,
        title="AI Assignment 1 - ConvNeXt V2 on CIFAR-10",
        author="AI-assisted submission for CISC3024 Pattern Recognition",
        subject="Recent Deep CNN application in computer vision and pattern recognition",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="all", frames=frame, onPage=draw_page))
    doc.build(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
