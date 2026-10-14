import json
from pathlib import Path
from typing import Dict, Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "treatments.json"

DEFAULT_TREATMENT = {
    "crop": "General Crop",
    "status": "Infected",
    "severity": "Moderate",
    "recommendation": "Isolate affected plants, prune diseased foliage, and consult local extension guidelines for appropriate treatment.",
    "preventative_measures": [
        "Avoid overhead irrigation",
        "Maintain adequate crop spacing",
        "Disinfect pruning tools after each cut"
    ],
    "organic_options": "Apply neem oil or organic copper-based fungicides."
}


class TreatmentService:
    def __init__(self, data_path: Path = DATA_PATH):
        self.treatments: Dict[str, Dict[str, Any]] = {}
        self._load_data(data_path)

    def _load_data(self, data_path: Path):
        try:
            if data_path.exists():
                with open(data_path, "r", encoding="utf-8") as f:
                    self.treatments = json.load(f)
            else:
                self.treatments = {}
        except Exception as e:
            print(f"Warning: Could not load treatments file: {e}")
            self.treatments = {}

    def get_recommendation(self, disease_name: str) -> Dict[str, Any]:
        """
        Retrieves treatment recommendation and preventative guidelines for a disease.
        """
        return self.treatments.get(disease_name, DEFAULT_TREATMENT)


# Singleton instance for easy reuse
_treatment_service = TreatmentService()


def get_treatment_service() -> TreatmentService:
    return _treatment_service
