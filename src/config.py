import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------
# DATA DIRECTORIES
# --------------------------------------------------

DATASET_DIR = BASE_DIR / "dataset"

RAW_DATA_DIR = DATASET_DIR / "raw"

PROCESSED_DATA_DIR = DATASET_DIR / "processed"

SPLIT_DATA_DIR = DATASET_DIR / "split"

TRAIN_DIR = SPLIT_DATA_DIR / "train"

VALIDATION_DIR = SPLIT_DATA_DIR / "validation"

TEST_DIR = SPLIT_DATA_DIR / "test"

# --------------------------------------------------
# MODEL DIRECTORY
# --------------------------------------------------

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "saree_design_resnet18.pth"

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

RESULTS_DIR = BASE_DIR / "results"

PLOTS_DIR = RESULTS_DIR / "plots"

METRICS_DIR = RESULTS_DIR / "metrics"

CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrix"

# --------------------------------------------------
# SAMPLE IMAGES
# --------------------------------------------------

SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"

# --------------------------------------------------
# IMAGE SETTINGS
# --------------------------------------------------

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))

# --------------------------------------------------
# TRAINING SETTINGS
# --------------------------------------------------

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 15))

LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0001))

NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))

NUM_CLASSES = int(os.getenv("NUM_CLASSES", 5))

CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", 0.70)
)

# --------------------------------------------------
# RANDOM SEED
# --------------------------------------------------

RANDOM_SEED = 42

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")