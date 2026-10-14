import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from config import DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, OUTPUTS_DIR

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_split(split_path: Path):
    counts = {}
    if not split_path.exists():
        return counts
    for cls_dir in sorted(split_path.iterdir()):
        if cls_dir.is_dir():
            files = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
            counts[cls_dir.name] = len(files)
    return counts

def main():
    train_counts = count_split(TRAIN_DIR)
    val_counts = count_split(VAL_DIR)
    test_counts = count_split(TEST_DIR)

    all_classes = sorted(list(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())))

    # Format human readable labels
    def format_label(name):
        return (name.replace("___", " - ")
                    .replace("__", " ")
                    .replace("_", " ")
                    .strip())

    records = []
    for cls in all_classes:
        tr = train_counts.get(cls, 0)
        va = val_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        tot = tr + va + te
        crop = cls.split("_")[0]
        is_healthy = "healthy" in cls.lower()
        records.append({
            "raw_class": cls,
            "label": format_label(cls),
            "crop": crop,
            "condition": "Healthy" if is_healthy else "Diseased",
            "train": tr,
            "val": va,
            "test": te,
            "total": tot
        })

    df = pd.DataFrame(records).sort_values(by="total", ascending=False).reset_index(drop=True)
    total_samples = df["total"].sum()

    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'DejaVu Sans'

    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    # Palette
    color_train = "#10b981"  # Emerald
    color_val = "#f59e0b"    # Amber
    color_test = "#3b82f6"   # Blue
    crop_palette = {"Pepper": "#ec4899", "Potato": "#8b5cf6", "Tomato": "#ef4444"}

    # -------------------------------------------------------------
    # Panel 1: Total Sample Count per Class (Horizontal Bar Chart)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    y_pos = np.arange(len(df))
    bars = ax1.barh(y_pos, df["total"], color="#0ea5e9", edgecolor="#0284c7", height=0.62, alpha=0.9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df["label"], fontsize=11, fontweight="bold", color="#1e293b")
    ax1.invert_yaxis()  # Highest on top
    ax1.set_xlabel("Number of Images", fontsize=12, fontweight="bold", color="#334155", labelpad=8)
    ax1.set_title("Total Dataset Distribution by Class (9,127 Total Images across 8 Classes)", 
                  fontsize=14, fontweight="bold", pad=12, color="#0f172a")
    ax1.grid(axis='x', linestyle='--', alpha=0.6)

    # Add data labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        width = bar.get_width()
        pct = (width / total_samples) * 100
        ax1.text(width + 25, bar.get_y() + bar.get_height()/2, 
                 f"{int(width):,} ({pct:.1f}%)", 
                 va='center', ha='left', fontsize=10.5, fontweight='bold', color="#0f172a")

    ax1.set_xlim(0, max(df["total"]) * 1.15)

    # -------------------------------------------------------------
    # Panel 2: Train / Val / Test Split Breakdown per Class
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    y_pos2 = np.arange(len(df))
    
    b1 = ax2.barh(y_pos2, df["train"], color=color_train, height=0.55, label="Train (~70%)", alpha=0.9)
    b2 = ax2.barh(y_pos2, df["val"], left=df["train"], color=color_val, height=0.55, label="Validation (~15%)", alpha=0.9)
    b3 = ax2.barh(y_pos2, df["test"], left=df["train"] + df["val"], color=color_test, height=0.55, label="Test (~15%)", alpha=0.9)

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(df["label"], fontsize=9.5, color="#1e293b")
    ax2.invert_yaxis()
    ax2.set_xlabel("Images per Split", fontsize=11, fontweight="bold", color="#334155")
    ax2.set_title("Dataset Split Breakdown (Train / Val / Test)", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax2.grid(axis='x', linestyle='--', alpha=0.6)

    # -------------------------------------------------------------
    # Panel 3: Distribution by Crop Type & Health Status
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    crop_counts = df.groupby("crop")["total"].sum()
    crops = crop_counts.index.tolist()
    
    healthy_counts = [df[(df["crop"] == c) & (df["condition"] == "Healthy")]["total"].sum() for c in crops]
    diseased_counts = [df[(df["crop"] == c) & (df["condition"] == "Diseased")]["total"].sum() for c in crops]

    x_crop = np.arange(len(crops))
    width = 0.35

    ax3.bar(x_crop - width/2, diseased_counts, width, label="Diseased", color="#f43f5e", alpha=0.9, edgecolor="#be123c")
    ax3.bar(x_crop + width/2, healthy_counts, width, label="Healthy", color="#10b981", alpha=0.9, edgecolor="#047857")

    ax3.set_xticks(x_crop)
    ax3.set_xticklabels([f"{c} ({crop_counts[c]:,} imgs)" for c in crops], fontsize=10.5, fontweight="bold", color="#1e293b")
    ax3.set_ylabel("Image Count", fontsize=11, fontweight="bold", color="#334155")
    ax3.set_title("Crop Aggregation (Diseased vs Healthy)", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
    ax3.legend(loc="upper left", frameon=True, fontsize=10)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)

    # Annotate bars in Panel 3
    for i in range(len(crops)):
        ax3.text(i - width/2, diseased_counts[i] + 30, f"{diseased_counts[i]:,}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#be123c")
        ax3.text(i + width/2, healthy_counts[i] + 30, f"{healthy_counts[i]:,}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#047857")

    ax3.set_ylim(0, max(diseased_counts) * 1.18)

    # Global title & branding
    plt.suptitle("CropMedic: Original Dataset Class Distribution & Split Analysis", 
                 fontsize=16, fontweight="bold", color="#0f172a", y=0.98)

    output_path = OUTPUTS_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated class distribution image at: {output_path}")

if __name__ == "__main__":
    main()
