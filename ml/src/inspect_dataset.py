import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, OUTPUTS_DIR

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_split(split_name: str, split_path: Path):
    if not split_path.exists():
        print(f"Error: {split_name} directory not found at {split_path}")
        return {}, [], []

    class_counts = {}
    corrupted_files = []
    image_shapes = []
    sample_images = {}

    classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])

    for cls in classes:
        cls_dir = split_path / cls
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        class_counts[cls] = len(files)

        for i, file_path in enumerate(files):
            try:
                with Image.open(file_path) as img:
                    img.verify()
                if i < 5:  # Sample dimensions
                    with Image.open(file_path) as img:
                        image_shapes.append(img.size)
                if cls not in sample_images:
                    sample_images[cls] = file_path
            except Exception as e:
                corrupted_files.append((file_path, str(e)))

    return class_counts, corrupted_files, image_shapes, sample_images


def main():
    print("=" * 60)
    print("STEP 1: INSPECTING CROPMEDIC DATASET")
    print(f"Dataset root: {DATASET_DIR}")
    print("=" * 60)

    train_counts, train_corrupt, train_shapes, sample_imgs = inspect_split("TRAIN", TRAIN_DIR)
    val_counts, val_corrupt, _, _ = inspect_split("VALIDATION", VAL_DIR)
    test_counts, test_corrupt, _, _ = inspect_split("TEST", TEST_DIR)

    all_classes = sorted(list(train_counts.keys()))
    print(f"\nTotal Classes Detected: {len(all_classes)}")
    assert len(all_classes) == 8, f"Expected 8 classes, found {len(all_classes)}"

    print("\n" + "-" * 60)
    print(f"{'Class Name':<35} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("-" * 60)

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())
    grand_total = total_train + total_val + total_test

    for cls in all_classes:
        tr = train_counts.get(cls, 0)
        va = val_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        tot = tr + va + te
        print(f"{cls:<35} | {tr:<7} | {va:<7} | {te:<7} | {tot:<7}")

    print("-" * 60)
    print(f"{'TOTAL':<35} | {total_train:<7} | {total_val:<7} | {total_test:<7} | {grand_total:<7}")
    print(f"Split Proportions: Train = {total_train/grand_total*100:.1f}%, Val = {total_val/grand_total*100:.1f}%, Test = {total_test/grand_total*100:.1f}%")

    # Image shape summary
    if train_shapes:
        shapes_set = set(train_shapes)
        print(f"\nSample Image Dimensions (W x H): {shapes_set}")

    # Corrupted check
    all_corrupt = train_corrupt + val_corrupt + test_corrupt
    if all_corrupt:
        print(f"\nWARNING: Found {len(all_corrupt)} corrupted files:")
        for path, err in all_corrupt[:5]:
            print(f"  - {path}: {err}")
    else:
        print("\nImage Integrity Check: PASSED (0 corrupted images found)")

    # Plot 1: Class Distribution
    print("\nGenerating class distribution visualization...")
    plt.figure(figsize=(12, 6))
    x = np.arange(len(all_classes))
    width = 0.25

    readable_labels = [c.replace("___", " - ").replace("__", " ").replace("_", " ") for c in all_classes]

    plt.bar(x - width, [train_counts.get(c, 0) for c in all_classes], width, label="Train (70%)", color="#15803d")
    plt.bar(x, [val_counts.get(c, 0) for c in all_classes], width, label="Val (15%)", color="#facc15")
    plt.bar(x + width, [test_counts.get(c, 0) for c in all_classes], width, label="Test (15%)", color="#3b82f6")

    plt.xlabel("Crop Disease Classes", fontweight="bold")
    plt.ylabel("Number of Images", fontweight="bold")
    plt.title("CropMedic Dataset Split & Class Distribution (8 Classes)", fontweight="bold", fontsize=13)
    plt.xticks(x, readable_labels, rotation=35, ha="right", fontsize=9)
    plt.legend()
    plt.tight_layout()
    dist_path = OUTPUTS_DIR / "class_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print(f"Saved: {dist_path}")

    # Plot 2: Sample Images
    print("Generating sample image grid...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, cls in enumerate(all_classes):
        img_path = sample_imgs.get(cls)
        if img_path:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            readable = cls.replace("___", "\n").replace("__", " ").replace("_", " ")
            axes[idx].set_title(readable, fontsize=10, fontweight="bold")
            axes[idx].axis("off")

    plt.suptitle("Sample Images for 8 Crop Disease Classes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    sample_path = OUTPUTS_DIR / "sample_images.png"
    plt.savefig(sample_path, dpi=200)
    plt.close()
    print(f"Saved: {sample_path}")

    print("\nDataset inspection complete!")


if __name__ == "__main__":
    main()
