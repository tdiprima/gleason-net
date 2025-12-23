"""
Configuration for Prostate Cancer Gleason Grading Trainer
"""

# Data paths
DATA_ROOT = "./patches_prostate_seer_john_6classes"

# Model settings
NUM_CLASSES = 4  # Benign, Gleason 3, Gleason 4, Gleason 5
INPUT_SIZE = 224  # ResNet50 expected input size
PRETRAINED = True  # Use ImageNet pretrained weights

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0001  # Lower LR for fine-tuning pretrained model
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 25

# Data loading optimization (72 cores available)
NUM_WORKERS = 16  # Good balance for 72 cores with GPU training
PIN_MEMORY = True
NON_BLOCKING = True
PREFETCH_FACTOR = 4

# Train/validation split
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Labels to use (skip 4 and 5)
VALID_LABELS = {0, 1, 2, 3}

# Label mapping (original -> training)
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

# Class names for display
CLASS_NAMES = ["Benign", "Gleason 3", "Gleason 4", "Gleason 5"]

# Device
DEVICE = "cuda"
