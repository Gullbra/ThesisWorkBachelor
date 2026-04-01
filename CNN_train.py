"""
Steganalysis CNN Training Script
=================================
Trains a CNN to classify images as cover (clean) or stego (embedded).

Folder structure expected:
    images/
    ├── train/
    │   ├── cover/   (7000 images)
    │   └── stego/   (7000 images)
    ├── val/
    │   ├── cover/   (1000 images)
    │   └── stego/   (1000 images)
    └── test/
        ├── cover/   (2000 images)
        └── stego/   (2000 images)

Usage:
    python stego_cnn_train.py
"""

import os
import sys
import time
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION / HYPERPARAMETERS                      ║
# ║              Edit these variables to tweak training behaviour               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# -- Paths -------------------------------------------------------------------
DATA_ROOT = "images"                   # Root folder containing train/val/test
OUTPUT_DIR = "outputs"                 # Where to save model, plots, logs
MODEL_FILENAME = "stego_cnn_best.pth"  # Saved model weights filename

# -- Training ----------------------------------------------------------------
NUM_EPOCHS = 20                # Total training epochs
BATCH_SIZE = 32                # Batch size for all loaders
LEARNING_RATE = 1e-3           # Initial learning rate
WEIGHT_DECAY = 1e-4            # L2 regularisation strength
OPTIMISER = "adam"             # "adam" | "adamw" | "sgd"
SGD_MOMENTUM = 0.9             # Only used when OPTIMISER="sgd"

# -- Learning‑rate scheduler -------------------------------------------------
USE_SCHEDULER = True           # Enable/disable LR scheduling
SCHEDULER_TYPE = "cosine"      # "cosine" | "step" | "plateau"
STEP_LR_STEP_SIZE = 10         # For StepLR: drop LR every N epochs
STEP_LR_GAMMA = 0.5            # For StepLR: multiply LR by this factor
PLATEAU_PATIENCE = 5           # For ReduceLROnPlateau: epochs to wait
PLATEAU_FACTOR = 0.5           # For ReduceLROnPlateau: LR reduction factor

# -- Early stopping ----------------------------------------------------------
USE_EARLY_STOPPING = True      # Enable/disable early stopping
EARLY_STOP_PATIENCE = 10       # Epochs without val improvement before stop
EARLY_STOP_MIN_DELTA = 1e-4   # Minimum change to count as improvement

# -- Data / augmentation -----------------------------------------------------
IMAGE_SIZE = 256               # Resize images to IMAGE_SIZE x IMAGE_SIZE
NUM_WORKERS = 4                # DataLoader worker processes
PIN_MEMORY = True              # Pin memory for faster GPU transfer
USE_AUGMENTATION = True        # Enable training‑time augmentation

# -- Model architecture ------------------------------------------------------
MODEL_TYPE = "stego_cnn"       # "stego_cnn" (custom) | "resnet18" (pretrained)
DROPOUT_RATE = 0.5             # Dropout probability in FC layers
BASE_FILTERS = 32              # Starting number of conv filters (custom CNN)

# -- Misc --------------------------------------------------------------------
RANDOM_SEED = 42               # Reproducibility seed
DEVICE = "auto"                # "auto" | "cuda" | "cpu"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              END OF CONFIG                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════════
#  TERMINAL DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class C:
    """ANSI colour codes for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Foreground
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    # Background
    BG_GREEN  = "\033[42m"
    BG_RED    = "\033[41m"
    BG_BLUE   = "\033[44m"
    BG_YELLOW = "\033[43m"


def term_width() -> int:
    """Get terminal width, default 80."""
    return shutil.get_terminal_size((80, 24)).columns


def banner(text: str, colour: str = C.CYAN, char: str = "═"):
    """Print a full-width banner."""
    w = term_width()
    pad = max(0, w - len(text) - 4)
    left = pad // 2
    right = pad - left
    print(f"\n{colour}{C.BOLD}{char * left}  {text}  {char * right}{C.RESET}")


def sub_banner(text: str, colour: str = C.BLUE, char: str = "─"):
    w = term_width()
    pad = max(0, w - len(text) - 4)
    left = pad // 2
    right = pad - left
    print(f"{colour}{char * left}  {text}  {char * right}{C.RESET}")


def kv(key: str, value, key_colour: str = C.GREY, val_colour: str = C.WHITE):
    """Print a key-value pair with aligned formatting."""
    print(f"  {key_colour}{key:<28}{C.RESET} {val_colour}{value}{C.RESET}")


def status(icon: str, msg: str, colour: str = C.GREEN):
    print(f"  {colour}{icon}{C.RESET}  {msg}")


def progress_bar(current: int, total: int, width: int = 30,
                 prefix: str = "", loss: float = None, acc: float = None) -> str:
    """Build an inline progress bar string."""
    frac = current / total if total > 0 else 0
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100

    extras = ""
    if loss is not None:
        extras += f"  loss {C.YELLOW}{loss:.4f}{C.RESET}"
    if acc is not None:
        extras += f"  acc {C.GREEN}{acc:.4f}{C.RESET}"

    return f"\r  {prefix} {C.CYAN}│{bar}│{C.RESET} {pct:5.1f}%  [{current}/{total}]{extras}"


def format_time(seconds: float) -> str:
    """Human-readable elapsed time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def spark_line(values: list, length: int = 20) -> str:
    """Tiny inline sparkline from a list of floats."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    # Take the last `length` values
    recent = values[-length:]
    return "".join(blocks[min(int((v - mn) / rng * 8), 8)] for v in recent)


def print_epoch_summary(epoch, num_epochs, train_loss, train_acc,
                        val_loss, val_acc, val_auc, lr, elapsed,
                        best_val_loss, improved, epochs_no_improve, history):
    """Rich per-epoch summary block."""
    w = term_width()

    # Epoch header with improvement indicator
    if improved:
        marker = f"{C.GREEN}{C.BOLD}★ NEW BEST{C.RESET}"
    elif epochs_no_improve > 0:
        patience_left = EARLY_STOP_PATIENCE - epochs_no_improve
        if patience_left <= 3:
            marker = f"{C.RED}⚠ patience {patience_left}/{EARLY_STOP_PATIENCE}{C.RESET}"
        else:
            marker = f"{C.YELLOW}○ patience {patience_left}/{EARLY_STOP_PATIENCE}{C.RESET}"
    else:
        marker = ""

    print()
    print(f"  {C.BOLD}{C.WHITE}Epoch {epoch}/{num_epochs}{C.RESET}  "
          f"{C.DIM}({format_time(elapsed)}){C.RESET}  {marker}")
    print(f"  {C.GREY}{'─' * min(60, w - 4)}{C.RESET}")

    # Train vs Val comparison table
    # Loss comparison with delta arrow
    loss_delta = val_loss - train_loss
    loss_arrow = f"{C.RED}▲{C.RESET}" if loss_delta > 0.05 else f"{C.GREEN}≈{C.RESET}"

    print(f"  {C.GREY}{'':>12}{'Train':>12}{'Val':>12}{'Gap':>12}{C.RESET}")

    # Loss row
    gap_loss = val_loss - train_loss
    gap_col = C.RED if gap_loss > 0.1 else (C.YELLOW if gap_loss > 0.03 else C.GREEN)
    print(f"  {'Loss':<12}"
          f"{C.YELLOW}{train_loss:>12.4f}{C.RESET}"
          f"{C.YELLOW}{val_loss:>12.4f}{C.RESET}"
          f"{gap_col}{gap_loss:>+12.4f}{C.RESET}")

    # Accuracy row
    gap_acc = val_acc - train_acc
    gap_col = C.RED if gap_acc < -0.05 else (C.YELLOW if gap_acc < -0.02 else C.GREEN)
    print(f"  {'Accuracy':<12}"
          f"{C.GREEN}{train_acc:>12.4f}{C.RESET}"
          f"{C.GREEN}{val_acc:>12.4f}{C.RESET}"
          f"{gap_col}{gap_acc:>+12.4f}{C.RESET}")

    # AUC row (val only)
    auc_colour = C.GREEN if val_auc > 0.9 else (C.YELLOW if val_auc > 0.7 else C.RED)
    print(f"  {'AUC':<12}{'—':>12}"
          f"{auc_colour}{val_auc:>12.4f}{C.RESET}"
          f"{'':>12}")

    # LR
    print(f"  {'LR':<12}{'':>12}{C.MAGENTA}{lr:>12.2e}{C.RESET}")

    # Mini sparklines for trends
    if len(history["val_loss"]) > 1:
        print()
        print(f"  {C.GREY}Val Loss trend :{C.RESET}  {C.CYAN}{spark_line(history['val_loss'])}{C.RESET}")
        print(f"  {C.GREY}Val Acc  trend :{C.RESET}  {C.GREEN}{spark_line(history['val_acc'])}{C.RESET}")
        print(f"  {C.GREY}Val AUC  trend :{C.RESET}  {C.MAGENTA}{spark_line(history['val_auc'])}{C.RESET}")

    # Best val loss reminder
    print(f"\n  {C.DIM}Best val_loss so far: {best_val_loss:.4f}{C.RESET}")


def print_config_table():
    """Print all hyperparameters in a neat grouped table."""
    banner("HYPERPARAMETER CONFIGURATION", C.MAGENTA)

    sub_banner("Paths", C.GREY)
    kv("DATA_ROOT", DATA_ROOT)
    kv("OUTPUT_DIR", OUTPUT_DIR)
    kv("MODEL_FILENAME", MODEL_FILENAME)

    sub_banner("Training", C.GREY)
    kv("NUM_EPOCHS", NUM_EPOCHS)
    kv("BATCH_SIZE", BATCH_SIZE)
    kv("LEARNING_RATE", f"{LEARNING_RATE:.1e}")
    kv("WEIGHT_DECAY", f"{WEIGHT_DECAY:.1e}")
    kv("OPTIMISER", OPTIMISER.upper())

    sub_banner("LR Scheduler", C.GREY)
    kv("USE_SCHEDULER", f"{'✓ Enabled' if USE_SCHEDULER else '✗ Disabled'}",
       val_colour=C.GREEN if USE_SCHEDULER else C.RED)
    if USE_SCHEDULER:
        kv("SCHEDULER_TYPE", SCHEDULER_TYPE)

    sub_banner("Early Stopping", C.GREY)
    kv("USE_EARLY_STOPPING", f"{'✓ Enabled' if USE_EARLY_STOPPING else '✗ Disabled'}",
       val_colour=C.GREEN if USE_EARLY_STOPPING else C.RED)
    if USE_EARLY_STOPPING:
        kv("EARLY_STOP_PATIENCE", EARLY_STOP_PATIENCE)
        kv("EARLY_STOP_MIN_DELTA", f"{EARLY_STOP_MIN_DELTA:.1e}")

    sub_banner("Data & Augmentation", C.GREY)
    kv("IMAGE_SIZE", f"{IMAGE_SIZE}×{IMAGE_SIZE}")
    kv("NUM_WORKERS", NUM_WORKERS)
    kv("USE_AUGMENTATION", f"{'✓ Enabled' if USE_AUGMENTATION else '✗ Disabled'}",
       val_colour=C.GREEN if USE_AUGMENTATION else C.RED)

    sub_banner("Model Architecture", C.GREY)
    kv("MODEL_TYPE", MODEL_TYPE)
    kv("DROPOUT_RATE", DROPOUT_RATE)
    kv("BASE_FILTERS", BASE_FILTERS)

    sub_banner("Misc", C.GREY)
    kv("RANDOM_SEED", RANDOM_SEED)
    kv("DEVICE", DEVICE)
    print()


def print_ascii_header():
    """Print a styled ASCII title block."""
    w = term_width()
    print(f"\n{C.CYAN}{C.BOLD}")
    lines = [
        "╔═══════════════════════════════════════════════════╗",
        "║        STEGANALYSIS  CNN  TRAINER  v2.0          ║",
        "║     Cover vs. Stego  ·  Binary Classification    ║",
        "╚═══════════════════════════════════════════════════╝",
    ]
    for line in lines:
        pad = (w - len(line)) // 2
        print(" " * max(0, pad) + line)
    print(C.RESET)
    print(f"  {C.DIM}Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ML CODE
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


# ---------------------------------------------------------------------------
# High‑pass residual preprocessing
# ---------------------------------------------------------------------------

class SRMFilterLayer(nn.Module):
    """
    Applies a bank of fixed high‑pass SRM (Spatial Rich Model) filters as a
    non‑trainable first layer.
    """

    def __init__(self):
        super().__init__()
        q = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=np.float32) / 12.0

        h = np.array([
            [-1, 2, -2, 2, -1],
            [ 2,-6,  8,-6,  2],
            [-2, 8,-12, 8, -2],
            [ 2,-6,  8,-6,  2],
            [-1, 2, -2, 2, -1],
        ], dtype=np.float32) / 12.0

        s = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32)

        filters = []
        for kern in [q, h, s]:
            for _ in range(3):
                f = np.zeros((3, 5, 5), dtype=np.float32)
                f[_, :, :] = kern
                filters.append(f)
        filters = np.stack(filters, axis=0)

        self.conv = nn.Conv2d(3, 9, kernel_size=5, padding=2, bias=False)
        self.conv.weight = nn.Parameter(
            torch.from_numpy(filters), requires_grad=False
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Custom CNN
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class StegoCNN(nn.Module):
    def __init__(self, base_filters=32, dropout=0.5):
        super().__init__()
        f = base_filters

        self.prep = SRMFilterLayer()

        self.features = nn.Sequential(
            ConvBlock(9,     f),
            ConvBlock(f,     f * 2),
            ConvBlock(f * 2, f * 4),
            ConvBlock(f * 4, f * 8),
            ConvBlock(f * 8, f * 8),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18(dropout=0.5) -> nn.Module:
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, 2),
    )
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_transforms():
    normalise = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if USE_AUGMENTATION:
        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalise,
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalise,
    ])

    return train_tf, eval_tf


def get_dataloaders():
    train_tf, eval_tf = get_transforms()

    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=eval_tf)

    banner("DATASET SUMMARY", C.BLUE)
    kv("Classes", train_ds.classes)
    kv("Class → idx", train_ds.class_to_idx)
    print()

    # Visual bar chart of split sizes
    splits = {"Train": len(train_ds), "Val": len(val_ds), "Test": len(test_ds)}
    max_count = max(splits.values())
    bar_max = 40
    for name, count in splits.items():
        bar_len = int(count / max_count * bar_max)
        bar = "█" * bar_len
        colour = C.GREEN if name == "Train" else (C.YELLOW if name == "Val" else C.CYAN)
        print(f"  {colour}{name:<6}{C.RESET} {colour}{bar}{C.RESET} {C.WHITE}{count:,}{C.RESET}")
    total = sum(splits.values())
    print(f"  {C.DIM}{'─' * 50}")
    print(f"  Total: {total:,} images{C.RESET}")
    print()

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    kv("Batches per epoch (train)", len(train_loader))
    kv("Batches per epoch (val)", len(val_loader))
    kv("Batches per epoch (test)", len(test_loader))
    print()

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training & evaluation loops (with progress bars)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimiser, device, epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    num_batches = len(loader)

    for batch_idx, (imgs, labels) in enumerate(loader, 1):
        imgs, labels = imgs.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        # Live progress bar
        avg_loss = running_loss / total
        avg_acc = correct / total
        bar = progress_bar(batch_idx, num_batches,
                           prefix=f"{C.BLUE}Train{C.RESET}",
                           loss=avg_loss, acc=avg_acc)
        sys.stdout.write(bar)
        sys.stdout.flush()

    sys.stdout.write("\n")
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, label="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    num_batches = len(loader)

    for batch_idx, (imgs, labels) in enumerate(loader, 1):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        # Live progress bar
        avg_loss = running_loss / total
        avg_acc = correct / total
        colour = C.YELLOW if label == "Val" else C.CYAN
        bar = progress_bar(batch_idx, num_batches,
                           prefix=f"{colour}{label:<5}{C.RESET}",
                           loss=avg_loss, acc=avg_acc)
        sys.stdout.write(bar)
        sys.stdout.flush()

    sys.stdout.write("\n")

    auc = roc_auc_score(all_labels, all_probs)
    return running_loss / total, correct / total, auc, all_labels, all_probs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_auc"], label="Val AUC", color="green")
    axes[2].set_title("Validation AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    status("📊", f"Training curves saved → {path}")


def save_roc_curve(labels, probs, output_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    status("📈", f"ROC curve saved → {path}")


def save_confusion_matrix(labels, preds, class_names, output_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.colorbar(im)
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    status("🔲", f"Confusion matrix saved → {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    global_start = time.time()

    # ── Header ──────────────────────────────────────────────────────────────
    print_ascii_header()
    print_config_table()

    set_seed(RANDOM_SEED)
    device = get_device()

    banner("ENVIRONMENT", C.GREEN)
    kv("Device", device)
    if device.type == "cuda":
        kv("GPU Name", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        kv("GPU Memory", f"{mem:.1f} GB")
    kv("PyTorch", torch.__version__)
    kv("CUDA available", torch.cuda.is_available())
    kv("Seed", RANDOM_SEED)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()
    class_names = train_loader.dataset.classes

    # ── Model ───────────────────────────────────────────────────────────────
    banner("MODEL", C.MAGENTA)

    if MODEL_TYPE == "resnet18":
        model = build_resnet18(dropout=DROPOUT_RATE).to(device)
        status("🏗️", "Architecture: Pretrained ResNet-18 backbone")
    else:
        model = StegoCNN(base_filters=BASE_FILTERS, dropout=DROPOUT_RATE).to(device)
        status("🏗️", f"Architecture: Custom StegoCNN  (base_filters={BASE_FILTERS})")

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kv("Total parameters", f"{total_params:,}")
    kv("Trainable parameters", f"{train_params:,}")

    # Show layer breakdown
    print(f"\n  {C.DIM}Layer structure:{C.RESET}")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen_tag = f" {C.RED}(frozen){C.RESET}" if trainable == 0 and param_count > 0 else ""
        print(f"    {C.GREY}├─{C.RESET} {C.WHITE}{name:<16}{C.RESET} "
              f"{C.DIM}{param_count:>10,} params{C.RESET}{frozen_tag}")
    print()

    # ── Loss / Optimiser / Scheduler ────────────────────────────────────────
    banner("OPTIMISATION", C.YELLOW)

    criterion = nn.CrossEntropyLoss()
    kv("Loss function", "CrossEntropyLoss")

    if OPTIMISER == "adam":
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMISER == "adamw":
        opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMISER == "sgd":
        opt = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                        momentum=SGD_MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimiser: {OPTIMISER}")
    kv("Optimiser", OPTIMISER.upper())

    scheduler = None
    if USE_SCHEDULER:
        if SCHEDULER_TYPE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
        elif SCHEDULER_TYPE == "step":
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=STEP_LR_STEP_SIZE,
                                                  gamma=STEP_LR_GAMMA)
        elif SCHEDULER_TYPE == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", patience=PLATEAU_PATIENCE, factor=PLATEAU_FACTOR
            )
        kv("Scheduler", SCHEDULER_TYPE)
    else:
        kv("Scheduler", "None")
    print()

    # ── History tracking ────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auc": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)

    # ── Training ────────────────────────────────────────────────────────────
    banner("TRAINING", C.GREEN, char="═")
    est_batches = len(train_loader) + len(val_loader)
    print(f"  {C.DIM}Up to {NUM_EPOCHS} epochs  ·  "
          f"{len(train_loader)} train batches + {len(val_loader)} val batches per epoch  ·  "
          f"Early stop patience = {EARLY_STOP_PATIENCE}{C.RESET}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        sub_banner(f"Epoch {epoch}/{NUM_EPOCHS}", C.GREY, "·")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, opt, device, epoch, NUM_EPOCHS
        )
        val_loss, val_acc, val_auc, _, _ = evaluate(
            model, val_loader, criterion, device, label="Val"
        )

        elapsed = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Checkpoint best model
        improved = val_loss < best_val_loss - EARLY_STOP_MIN_DELTA
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        # Print rich summary
        print_epoch_summary(
            epoch, NUM_EPOCHS, train_loss, train_acc,
            val_loss, val_acc, val_auc, lr_now, elapsed,
            best_val_loss, improved, epochs_no_improve, history
        )

        if improved:
            status("💾", f"Model checkpoint saved → {best_model_path}", C.GREEN)

        # ETA estimate
        elapsed_total = time.time() - global_start
        avg_epoch_time = elapsed_total / epoch
        remaining_epochs = NUM_EPOCHS - epoch
        eta = avg_epoch_time * remaining_epochs
        print(f"  {C.DIM}ETA for remaining {remaining_epochs} epochs: "
              f"~{format_time(eta)}{C.RESET}")

        # Scheduler step
        if scheduler is not None:
            if SCHEDULER_TYPE == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping
        if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print()
            banner("⛔ EARLY STOPPING TRIGGERED", C.RED)
            print(f"  {C.RED}No improvement for {EARLY_STOP_PATIENCE} consecutive epochs.{C.RESET}")
            print(f"  {C.DIM}Best val_loss was {best_val_loss:.4f} "
                  f"(achieved at epoch {epoch - EARLY_STOP_PATIENCE}){C.RESET}")
            break

    total_train_time = time.time() - global_start
    print()
    banner(f"TRAINING COMPLETE  ·  {format_time(total_train_time)} total", C.GREEN)

    # ── Quick training summary ──────────────────────────────────────────────
    print(f"\n  {C.BOLD}Training Summary{C.RESET}")
    kv("Epochs completed", len(history["train_loss"]))
    kv("Best val_loss", f"{best_val_loss:.4f}")
    kv("Best val_acc", f"{max(history['val_acc']):.4f}")
    kv("Best val_auc", f"{max(history['val_auc']):.4f}")
    kv("Total time", format_time(total_train_time))
    print()

    # ── Test evaluation ─────────────────────────────────────────────────────
    banner("TEST SET EVALUATION", C.CYAN, char="═")
    print(f"  {C.DIM}Loading best checkpoint from {best_model_path}{C.RESET}\n")

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_loss, test_acc, test_auc, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device, label="Test"
    )
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]

    print()
    sub_banner("Test Results", C.CYAN)

    # Big result display
    acc_colour = C.GREEN if test_acc > 0.9 else (C.YELLOW if test_acc > 0.7 else C.RED)
    auc_colour = C.GREEN if test_auc > 0.9 else (C.YELLOW if test_auc > 0.7 else C.RED)

    print(f"""
  ┌─────────────────────────────────────────┐
  │  {C.BOLD}Test Loss{C.RESET}     {C.YELLOW}{test_loss:.4f}{C.RESET}                     │
  │  {C.BOLD}Test Accuracy{C.RESET} {acc_colour}{test_acc:.4f}{C.RESET}  {'✓' if test_acc > 0.9 else '○'}                  │
  │  {C.BOLD}Test AUC{C.RESET}      {auc_colour}{test_auc:.4f}{C.RESET}  {'✓' if test_auc > 0.9 else '○'}                  │
  └─────────────────────────────────────────┘
""")

    # Confusion matrix in terminal
    cm = confusion_matrix(test_labels, test_preds)
    sub_banner("Confusion Matrix", C.CYAN)
    print(f"""
  {C.DIM}Predicted →{C.RESET}    {C.WHITE}{class_names[0]:>10}  {class_names[1]:>10}{C.RESET}
  {C.WHITE}{class_names[0]:<12}{C.RESET}   {C.GREEN}{cm[0][0]:>10}{C.RESET}  {C.RED}{cm[0][1]:>10}{C.RESET}
  {C.WHITE}{class_names[1]:<12}{C.RESET}   {C.RED}{cm[1][0]:>10}{C.RESET}  {C.GREEN}{cm[1][1]:>10}{C.RESET}
""")

    # Classification report
    sub_banner("Classification Report", C.CYAN)
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # ── Save plots ──────────────────────────────────────────────────────────
    banner("SAVING OUTPUTS", C.MAGENTA)
    save_training_curves(history, OUTPUT_DIR)
    save_roc_curve(test_labels, test_probs, OUTPUT_DIR)
    save_confusion_matrix(test_labels, test_preds, class_names, OUTPUT_DIR)

    # ── Save run config ─────────────────────────────────────────────────────
    log_path = os.path.join(OUTPUT_DIR, "run_config.txt")
    with open(log_path, "w") as f:
        f.write("Steganalysis CNN — Run Configuration\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training time: {format_time(total_train_time)}\n\n")
        for var in [
            "NUM_EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY",
            "OPTIMISER", "USE_SCHEDULER", "SCHEDULER_TYPE",
            "USE_EARLY_STOPPING", "EARLY_STOP_PATIENCE",
            "IMAGE_SIZE", "MODEL_TYPE", "BASE_FILTERS", "DROPOUT_RATE",
            "USE_AUGMENTATION", "RANDOM_SEED",
        ]:
            f.write(f"{var} = {globals()[var]}\n")
        f.write(f"\nTest Accuracy = {test_acc:.4f}\n")
        f.write(f"Test AUC      = {test_auc:.4f}\n")
    status("📝", f"Run config saved → {log_path}")

    # ── Final sign-off ──────────────────────────────────────────────────────
    total_time = time.time() - global_start
    print()
    banner("ALL DONE", C.GREEN, char="═")
    print(f"""
  {C.GREEN}{C.BOLD}✓{C.RESET} Best model      → {C.WHITE}{best_model_path}{C.RESET}
  {C.GREEN}{C.BOLD}✓{C.RESET} Training curves → {C.WHITE}{OUTPUT_DIR}/training_curves.png{C.RESET}
  {C.GREEN}{C.BOLD}✓{C.RESET} ROC curve       → {C.WHITE}{OUTPUT_DIR}/roc_curve.png{C.RESET}
  {C.GREEN}{C.BOLD}✓{C.RESET} Confusion matrix→ {C.WHITE}{OUTPUT_DIR}/confusion_matrix.png{C.RESET}
  {C.GREEN}{C.BOLD}✓{C.RESET} Run config      → {C.WHITE}{OUTPUT_DIR}/run_config.txt{C.RESET}

  {C.DIM}Total wall time: {format_time(total_time)}{C.RESET}
""")


if __name__ == "__main__":
    main()
