import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

from config import MODEL_SAVE_PATH, CLASS_NAMES_PATH, IMAGE_SIZE


class TrainedDiseaseClassifier:
    """
    CNN Disease Classifier using trained MobileNetV2 architecture.
    Drop-in replacement for BaseDiseaseClassifier / MockDiseaseClassifier in FastAPI backend.
    """

    def __init__(self, model_path: Path = MODEL_SAVE_PATH, class_names_path: Path = CLASS_NAMES_PATH):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_mapping = None
        self._load()

    def _load(self):
        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class mapping file not found at {self.class_names_path}")
        with open(self.class_names_path, "r", encoding="utf-8") as f:
            self.class_mapping = json.load(f)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path))

    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict(self, image_bytes: bytes, filename: str = ""):
        """
        Takes raw image bytes, returns (predicted_disease_name, confidence_score).
        Compatible with BaseDiseaseClassifier interface.
        """
        import io
        img = Image.open(io.BytesIO(image_bytes))
        img_batch = self.preprocess_image(img)

        preds = self.model.predict(img_batch, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])

        disease_name = self.class_mapping["index_to_readable"][str(top_idx)]
        return disease_name, round(confidence, 4)

    def predict_file(self, file_path: Path):
        """
        Convenience method for CLI testing of single image files.
        Returns top-3 predictions.
        """
        img = Image.open(file_path)
        img_batch = self.preprocess_image(img)

        preds = self.model.predict(img_batch, verbose=0)[0]
        top_indices = np.argsort(preds)[::-1][:3]

        results = []
        for idx in top_indices:
            results.append({
                "disease": self.class_mapping["index_to_readable"][str(idx)],
                "confidence": float(preds[idx]),
            })
        return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_leaf_image>")
        print("Example: python src/predict.py dataset/test/Tomato___Early_blight/0a0b6727-4638-42f7-872f-50d4f3b14093___RS_Erly.B 9481.JPG")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found at {image_path}")
        sys.exit(1)

    classifier = TrainedDiseaseClassifier()
    top_preds = classifier.predict_file(image_path)

    print("\n" + "=" * 50)
    print(f"DISEASE PREDICTION REPORT")
    print(f"Image File: {image_path.name}")
    print("=" * 50)
    print(f"Top Prediction : {top_preds[0]['disease']}")
    print(f"Confidence     : {top_preds[0]['confidence'] * 100:.2f}%")
    print("-" * 50)
    print("Top-3 Candidates:")
    for i, p in enumerate(top_preds, 1):
        print(f"  {i}. {p['disease']:<28} - {p['confidence'] * 100:.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
