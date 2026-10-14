import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from config import (
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    MODEL_SAVE_PATH,
    CLASS_NAMES_PATH,
    OUTPUTS_DIR,
)


def evaluate_model():
    print("=" * 60)
    print("STEP 10 & 11: EVALUATING MODEL ON UNTOUCHED TEST DATASET")
    print("=" * 60)

    # 1. Load class mapping
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    readable_classes = class_mapping["readable_classes"]
    raw_classes = class_mapping["raw_classes"]

    # 2. Load trained model
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_SAVE_PATH}. Run train.py first.")

    print(f"Loading trained model from: {MODEL_SAVE_PATH}")
    model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

    # 3. Load test dataset (untouched)
    print(f"Loading test dataset from: {TEST_DIR}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    # 4. Evaluate overall loss and accuracy
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print("\n" + "=" * 60)
    print(f"OVERALL TEST ACCURACY : {accuracy * 100:.2f}%")
    print(f"OVERALL TEST LOSS     : {loss:.4f}")
    print("=" * 60)

    # 5. Extract all true labels and predictions
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 6. Classification Report (Precision, Recall, F1)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=readable_classes,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=readable_classes,
        digits=4,
    )

    print("\nDETAILED CLASSIFICATION REPORT (PER-CLASS PERFORMANCE):")
    print("-" * 60)
    print(report_text)
    print("-" * 60)

    # Save classification report to JSON
    report_json_path = OUTPUTS_DIR / "classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved classification report to: {report_json_path}")

    # 7. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="YlGnBu",
        xticklabels=readable_classes,
        yticklabels=readable_classes,
        cbar=True,
    )
    plt.title(f"CropMedic CNN Confusion Matrix (Test Accuracy: {accuracy*100:.2f}%)", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # 8. Confusion Analysis
    print("\nNOTABLE CONFUSION ANALYSIS:")
    for i in range(len(readable_classes)):
        for j in range(len(readable_classes)):
            if i != j and cm[i][j] > 0:
                percent = (cm[i][j] / cm[i].sum()) * 100
                if percent >= 1.0:  # Report any misclassification >= 1%
                    print(f"  - True [{readable_classes[i]}] misclassified as [{readable_classes[j]}]: {cm[i][j]} images ({percent:.1f}%)")

    return accuracy, report_dict


if __name__ == "__main__":
    evaluate_model()
