import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path

from config import (
    TRAIN_DIR,
    VAL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
    SEED,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    MODEL_SAVE_PATH,
    CLASS_NAMES_PATH,
    OUTPUTS_DIR,
)

# Deterministic naming mapping for frontend & backend consistency
HUMAN_READABLE_NAMES = {
    "Pepper__bell___Bacterial_spot": "Pepper Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Potato Healthy",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___healthy": "Tomato Healthy",
}


def create_data_loaders():
    print(f"Loading training dataset from {TRAIN_DIR}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    print(f"Loading validation dataset from {VAL_DIR}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    raw_class_names = train_ds.class_names
    print(f"Detected Class Directories ({len(raw_class_names)}): {raw_class_names}")

    class_mapping = {
        "raw_classes": raw_class_names,
        "readable_classes": [HUMAN_READABLE_NAMES.get(c, c) for c in raw_class_names],
        "index_to_readable": {i: HUMAN_READABLE_NAMES.get(c, c) for i, c in enumerate(raw_class_names)},
        "index_to_raw": {i: c for i, c in enumerate(raw_class_names)},
    }

    # Save deterministic mapping
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Saved class mapping to: {CLASS_NAMES_PATH}")

    # Optimize data loading pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_mapping


def build_model():
    # Data Augmentation layer (active only during training)
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )

    # Base Pretrained Model: MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base model for Stage 1

    # Compose Functional Model
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_layer")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output_predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CropMedic_MobileNetV2")
    return model, base_model


def plot_training_history(history1, history2=None):
    acc = history1.history["accuracy"]
    val_acc = history1.history["val_accuracy"]
    loss = history1.history["loss"]
    val_loss = history1.history["val_loss"]

    if history2:
        acc += history2.history["accuracy"]
        val_acc += history2.history["val_accuracy"]
        loss += history2.history["loss"]
        val_loss += history2.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#15803d", lw=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#3b82f6", lw=2)
    if history2:
        plt.axvline(x=len(history1.history["accuracy"]), color="red", linestyle="--", label="Start Fine-Tuning")
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Accuracy", fontweight="bold")
    plt.title("Training vs Validation Accuracy", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#15803d", lw=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#3b82f6", lw=2)
    if history2:
        plt.axvline(x=len(history1.history["loss"]), color="red", linestyle="--", label="Start Fine-Tuning")
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Loss", fontweight="bold")
    plt.title("Training vs Validation Loss", fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = OUTPUTS_DIR / "training_history.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved training history curves to: {hist_path}")


def main():
    print("=" * 60)
    print("STEP 6 & 7 & 8: TRAINING CNN CLASSIFIER")
    print("=" * 60)

    # 1. Load data
    train_ds, val_ds, class_mapping = create_data_loaders()

    # 2. Build model
    model, base_model = build_model()
    model.summary()

    # 3. Stage 1: Train Top Layers (Feature Extraction)
    print("\n" + "=" * 60)
    print("STAGE 1: Training Custom Classification Head (Frozen Base)")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=callbacks_stage1,
    )

    # 4. Stage 2: Fine-Tuning (Unfreeze Top 30 layers of base model)
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-Tuning Top Layers of MobileNetV2")
    print("=" * 60)

    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Base model total layers: {len(base_model.layers)}")
    print(f"Fine-tuning layers from layer index {fine_tune_at} to {len(base_model.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        callbacks=callbacks_stage2,
    )

    # 5. Plot and save training history
    plot_training_history(history1, history2)

    # 6. Save final best model confirmation
    print(f"\nBest model successfully saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
