from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_SAVE_PATH = MODELS_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# Training Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8
SEED = 42

# Stage 1: Feature Extraction
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3

# Stage 2: Fine-Tuning
STAGE2_EPOCHS = 10
STAGE2_LR = 1e-5

# Ensure output and model folders exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
