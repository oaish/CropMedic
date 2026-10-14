from abc import ABC, abstractmethod
import io
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image
import numpy as np

logger = logging.getLogger("uvicorn.error")

# Candidate model paths
CURRENT_DIR = Path(__file__).resolve().parent
POSSIBLE_MODEL_PATHS = [
    CURRENT_DIR.parent.parent.parent / "ml" / "models" / "crop_disease_model.keras",
    CURRENT_DIR.parent / "models" / "crop_disease_model.keras",
    Path("C:/dev/web/CropMedic/ml/models/crop_disease_model.keras"),
]

POSSIBLE_CLASS_PATHS = [
    CURRENT_DIR.parent.parent.parent / "ml" / "models" / "class_names.json",
    CURRENT_DIR.parent / "models" / "class_names.json",
    Path("C:/dev/web/CropMedic/ml/models/class_names.json"),
]


class BaseDiseaseClassifier(ABC):
    """
    Abstract Base Class for Plant Disease Classifiers.
    """

    @abstractmethod
    def predict(self, image_bytes: bytes, filename: str = "") -> Tuple[str, float]:
        """
        Takes raw image bytes and optional filename, returns (predicted_disease_name, confidence_score).
        :return: (disease_name, confidence_score) where confidence is between 0.0 and 1.0.
        """
        pass


class MockDiseaseClassifier(BaseDiseaseClassifier):
    """
    Fallback mock classifier for prototyping when trained model is unavailable.
    """

    SUPPORTED_DISEASES = [
        "Tomato Early Blight",
        "Tomato Late Blight",
        "Tomato Healthy",
        "Potato Early Blight",
        "Potato Late Blight",
        "Pepper Bacterial Spot",
    ]

    def predict(self, image_bytes: bytes, filename: str = "") -> Tuple[str, float]:
        import hashlib
        if not image_bytes:
            raise ValueError("Empty image content provided.")

        lower_name = filename.lower()
        if "healthy" in lower_name:
            return "Tomato Healthy", 0.98
        elif "potato" in lower_name and "late" in lower_name:
            return "Potato Late Blight", 0.93
        elif "potato" in lower_name:
            return "Potato Early Blight", 0.91
        elif "pepper" in lower_name or "spot" in lower_name:
            return "Pepper Bacterial Spot", 0.89
        elif "late" in lower_name:
            return "Tomato Late Blight", 0.96
        else:
            hash_val = int(hashlib.md5(image_bytes).hexdigest(), 16)
            disease_idx = hash_val % len(self.SUPPORTED_DISEASES)
            selected_disease = self.SUPPORTED_DISEASES[disease_idx]
            confidence = round(0.86 + ((hash_val % 13) / 100.0), 2)
            return selected_disease, confidence


class TrainedDiseaseClassifier(BaseDiseaseClassifier):
    """
    Trained CNN Disease Classifier using MobileNetV2 architecture.
    """

    def __init__(self, model_path: Optional[Path] = None, class_names_path: Optional[Path] = None):
        self.model = None
        self.class_mapping = None
        self.image_size = (224, 224)

        # Resolve model path
        self.model_path = model_path
        if not self.model_path:
            for p in POSSIBLE_MODEL_PATHS:
                if p.exists():
                    self.model_path = p
                    break

        # Resolve class names path
        self.class_names_path = class_names_path
        if not self.class_names_path:
            for p in POSSIBLE_CLASS_PATHS:
                if p.exists():
                    self.class_names_path = p
                    break

        self._load_model_and_classes()

    def _load_model_and_classes(self):
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found in candidates: {POSSIBLE_MODEL_PATHS}")

        if not self.class_names_path or not self.class_names_path.exists():
            raise FileNotFoundError(f"Class names mapping not found in candidates: {POSSIBLE_CLASS_PATHS}")

        logger.info(f"Loading trained model from: {self.model_path}")
        with open(self.class_names_path, "r", encoding="utf-8") as f:
            self.class_mapping = json.load(f)

        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(self.model_path))
        logger.info(f"Successfully loaded trained CNN model with {len(self.class_mapping['readable_classes'])} classes.")

    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_resized = img.resize(self.image_size)
        img_array = np.array(img_resized, dtype=np.float32)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_bytes: bytes, filename: str = "") -> Tuple[str, float]:
        if not image_bytes:
            raise ValueError("Empty image content provided.")

        img = Image.open(io.BytesIO(image_bytes))
        img_batch = self.preprocess_image(img)

        preds = self.model.predict(img_batch, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])

        # Lookup readable class name
        disease_name = self.class_mapping["index_to_readable"][str(top_idx)]
        return disease_name, round(confidence, 4)


# Singleton instance initialization
_classifier_instance: Optional[BaseDiseaseClassifier] = None


def get_classifier() -> BaseDiseaseClassifier:
    """
    Dependency injection factory returning the trained CNN model (or mock fallback).
    """
    global _classifier_instance
    if _classifier_instance is None:
        try:
            _classifier_instance = TrainedDiseaseClassifier()
        except Exception as e:
            logger.warning(f"Could not load trained model ({e}). Falling back to MockDiseaseClassifier.")
            _classifier_instance = MockDiseaseClassifier()
    return _classifier_instance
