"""
Configuration for Prostate Cancer Gleason Grading Trainer
"""

import numpy as np

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

# ============================================================================
# CLASS WEIGHTING CONFIGURATION
# ============================================================================

# Raw class counts from your dataset (labels 0-3 only)
CLASS_COUNTS = np.array([19695, 26816, 8461, 815], dtype=np.float32)

# Weighting strategy options:
#   "none"              - No weighting (original behavior)
#   "inverse"           - Inverse frequency: total / (num_classes * count)
#   "inverse_sqrt"      - Square root of inverse frequency (softer)
#   "effective"         - Effective number of samples (Class-Balanced Loss paper)
#   "focal"             - Use Focal Loss instead of CrossEntropy
WEIGHTING_STRATEGY = "effective"  # <-- Change this to switch strategies

# For "effective" strategy: beta parameter (0.9, 0.99, 0.999, 0.9999 are common)
# Higher beta = more aggressive weighting toward minority classes
EFFECTIVE_BETA = 0.9999

# For "focal" loss: focusing parameter gamma
# gamma=0 equivalent to CE, gamma=2 is common, higher = more focus on hard examples
FOCAL_GAMMA = 2.0

# Optional: manually set weights (overrides computed weights if not None)
# Example: MANUAL_WEIGHTS = [1.0, 1.0, 2.0, 10.0]
MANUAL_WEIGHTS = None

# Whether to use weighted random sampling (alternative to loss weighting)
# If True, oversamples minority classes during training
USE_WEIGHTED_SAMPLING = False

# ============================================================================
# EARLY STOPPING CONFIGURATION
# ============================================================================

# Enable/disable early stopping
EARLY_STOPPING = True

# Number of epochs with no improvement before stopping
EARLY_STOPPING_PATIENCE = 5

# Minimum improvement to qualify as "improvement"
EARLY_STOPPING_MIN_DELTA = 0.0001

# Metric to monitor: "accuracy", "f1", "loss"
EARLY_STOPPING_METRIC = "f1"


def compute_class_weights(strategy, class_counts, beta=0.999):
    """
    Compute class weights based on the selected strategy.

    Args:
        strategy: One of "none", "inverse", "inverse_sqrt", "effective"
        class_counts: Array of sample counts per class
        beta: Parameter for effective number strategy

    Returns:
        numpy array of class weights, or None for "none"/"focal" strategies
    """
    if strategy == "none" or strategy == "focal":
        return None

    total_samples = class_counts.sum()
    num_classes = len(class_counts)

    if strategy == "inverse":
        # Standard inverse frequency weighting
        weights = total_samples / (num_classes * class_counts)

    elif strategy == "inverse_sqrt":
        # Softer weighting using square root
        weights = np.sqrt(total_samples / (num_classes * class_counts))

    elif strategy == "effective":
        # Effective number of samples (Class-Balanced Loss)
        # Paper: "Class-Balanced Loss Based on Effective Number of Samples"
        # https://arxiv.org/abs/1901.05555
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num

    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")

    # Normalize weights so they sum to num_classes (keeps loss scale similar)
    weights = weights / weights.sum() * num_classes

    return weights


def get_class_weights():
    """
    Get the class weights based on current configuration.

    Returns:
        numpy array of weights, or None if no weighting
    """
    if MANUAL_WEIGHTS is not None:
        return np.array(MANUAL_WEIGHTS, dtype=np.float32)

    return compute_class_weights(
        strategy=WEIGHTING_STRATEGY, 
        class_counts=CLASS_COUNTS, 
        beta=EFFECTIVE_BETA
    )


def print_weight_info():
    """Print information about the class weighting configuration."""
    print("\nClass Weighting Configuration:")
    print(f"  Strategy: {WEIGHTING_STRATEGY}")

    if WEIGHTING_STRATEGY == "effective":
        print(f"  Beta: {EFFECTIVE_BETA}")
    elif WEIGHTING_STRATEGY == "focal":
        print(f"  Gamma: {FOCAL_GAMMA}")

    weights = get_class_weights()

    print("\n  Class Distribution:")
    total = CLASS_COUNTS.sum()
    for i, (name, count) in enumerate(zip(CLASS_NAMES, CLASS_COUNTS)):
        pct = 100.0 * count / total
        weight_str = f", weight={weights[i]:.4f}" if weights is not None else ""
        print(f"    {name}: {int(count):,} ({pct:.1f}%){weight_str}")

    if USE_WEIGHTED_SAMPLING:
        print("\n  Weighted Random Sampling: ENABLED")
