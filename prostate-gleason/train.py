"""
Main training script for Prostate Cancer Gleason Grading
"""

import sys
import warnings
from time import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from config import (BATCH_SIZE, CLASS_COUNTS, CLASS_NAMES, DATA_ROOT, DEVICE,
                    EFFECTIVE_BETA, FOCAL_GAMMA, INPUT_SIZE, LABEL_MAP,
                    LEARNING_RATE, NON_BLOCKING, NUM_CLASSES, NUM_EPOCHS,
                    NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR, PRETRAINED,
                    RANDOM_SEED, TRAIN_SPLIT, USE_WEIGHTED_SAMPLING,
                    VALID_LABELS, WEIGHT_DECAY, WEIGHTING_STRATEGY,
                    EARLY_STOPPING, EARLY_STOPPING_PATIENCE,
                    EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_METRIC,
                    get_class_weights, print_weight_info)
from data import get_loaders
from eval import evaluate
from losses import get_criterion
from model import create_model

warnings.filterwarnings("ignore")


def print_system_info(device):
    """Display system and PyTorch configuration."""
    print("=" * 70)
    print("PROSTATE CANCER GLEASON GRADING - RESNET50 TRAINER")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: Using CPU - Training will be slow!")

    print(f"Training Device: {device}")
    print("=" * 70)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    Returns average loss and accuracy.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_batches = len(loader)
    log_interval = max(1, num_batches // 5)

    for batch_idx, (data, target) in enumerate(loader):
        if data is None:
            continue

        # Move batch to device
        data = data.to(device, non_blocking=NON_BLOCKING)
        target = target.to(device, non_blocking=NON_BLOCKING)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        preds = output.argmax(dim=1)
        correct = (preds == target).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += target.size(0)

        # Progress logging
        if (batch_idx + 1) % log_interval == 0:
            current_acc = 100.0 * total_correct / total_samples
            print(
                f"  Batch {batch_idx + 1}/{num_batches} | "
                f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%"
            )

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def plot_confusion_matrix(cm, class_names):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Prostate Gleason Grading")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'max' for metrics like accuracy/f1, 'min' for loss
    """
    
    def __init__(self, patience=5, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, current_value):
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
            
        Returns:
            True if this is a new best, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return True
        
        if self.mode == 'max':
            improved = current_value > self.best_value + self.min_delta
        else:  # mode == 'min'
            improved = current_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def main():
    """Main training function."""
    # Redirect stdout to file
    log_file = open("training_log.txt", "w")
    original_stdout = sys.stdout
    sys.stdout = log_file

    try:
        # Device setup
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        print_system_info(device)

        # Set random seed for reproducibility
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)

        # Print configuration
        print("\nConfiguration:")
        print(f"  Data Root: {DATA_ROOT}")
        print(f"  Number of Classes: {NUM_CLASSES}")
        print(f"  Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Learning Rate: {LEARNING_RATE}")
        print(f"  Weight Decay: {WEIGHT_DECAY}")
        print(f"  Epochs: {NUM_EPOCHS}")
        print(f"  Workers: {NUM_WORKERS}")
        print(f"  Pin Memory: {PIN_MEMORY}")
        print(f"  Pretrained: {PRETRAINED}")
        print(f"  Valid Labels: {VALID_LABELS}")

        # Print class weighting info
        print_weight_info()

        # Load data
        print("\n" + "=" * 70)
        print("LOADING DATASET")
        print("=" * 70)

        train_loader, val_loader = get_loaders(
            DATA_ROOT,
            BATCH_SIZE,
            INPUT_SIZE,
            VALID_LABELS,
            LABEL_MAP,
            TRAIN_SPLIT,
            RANDOM_SEED,
            NUM_WORKERS,
            PIN_MEMORY,
            PREFETCH_FACTOR,
            use_weighted_sampling=USE_WEIGHTED_SAMPLING,
            num_classes=NUM_CLASSES,
        )

        if train_loader is None or val_loader is None:
            print("Failed to load dataset. Exiting.")
            return

        # Initialize model
        print("\n" + "=" * 70)
        print("INITIALIZING MODEL")
        print("=" * 70)

        model = create_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED)
        model = model.to(device)

        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        # Get class weights
        class_weights = get_class_weights()

        # Loss function with class weighting
        print("\n" + "=" * 70)
        print("LOSS FUNCTION")
        print("=" * 70)
        print(f"Strategy: {WEIGHTING_STRATEGY}")

        criterion = get_criterion(
            strategy=WEIGHTING_STRATEGY,
            class_weights=class_weights,
            class_counts=CLASS_COUNTS,
            num_classes=NUM_CLASSES,
            focal_gamma=FOCAL_GAMMA,
            effective_beta=EFFECTIVE_BETA,
        )

        # Move criterion to device (for FocalLoss buffers)
        criterion = criterion.to(device)

        if class_weights is not None:
            print(f"Class weights: {class_weights}")

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=NUM_EPOCHS, 
            eta_min=LEARNING_RATE * 0.01
        )

        # Training loop
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)

        # Early stopping setup
        if EARLY_STOPPING:
            mode = 'min' if EARLY_STOPPING_METRIC == 'loss' else 'max'
            early_stopper = EarlyStopping(
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                mode=mode
            )
            print(f"\nEarly Stopping: enabled")
            print(f"  Patience: {EARLY_STOPPING_PATIENCE} epochs")
            print(f"  Min delta: {EARLY_STOPPING_MIN_DELTA}")
            print(f"  Monitoring: {EARLY_STOPPING_METRIC}")

        start_time = time()
        best_test_acc = 0.0
        best_f1 = 0.0

        for epoch in range(NUM_EPOCHS):
            epoch_start = time()
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} (LR: {current_lr:.6f})")
            print("-" * 50)

            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

            # Evaluate
            acc, prec, rec, f1, cm = evaluate(model, val_loader, device)

            # Step scheduler
            scheduler.step()

            epoch_time = time() - epoch_start

            # Print epoch summary
            print(f"\ntrain_loss={train_loss:.4f} | train_acc={train_acc:.4f}")
            print(f"test_acc={acc:.4f} | precision={prec:.4f} | recall={rec:.4f} | f1={f1:.4f}")
            print(f"Epoch Time: {epoch_time:.1f}s")

            # Determine metric for early stopping and model saving
            if EARLY_STOPPING_METRIC == "f1":
                current_metric = f1
            elif EARLY_STOPPING_METRIC == "loss":
                current_metric = train_loss
            else:  # accuracy
                current_metric = acc

            # Early stopping check
            if EARLY_STOPPING:
                is_best = early_stopper(current_metric)
                
                if is_best:
                    best_test_acc = acc
                    best_f1 = f1
                    torch.save(model.state_dict(), "best_model.pth")
                    print(f"New best model ({EARLY_STOPPING_METRIC}={current_metric:.4f})")
                else:
                    epochs_left = EARLY_STOPPING_PATIENCE - early_stopper.counter
                    print(f"No improvement... (patience: {epochs_left}/{EARLY_STOPPING_PATIENCE})")
                
                if early_stopper.should_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
            else:
                # Original behavior without early stopping
                if acc > best_test_acc:
                    best_test_acc = acc
                    best_f1 = f1
                    torch.save(model.state_dict(), "best_model.pth")
                    print("New best model (by test accuracy)")
                else:
                    print("No improvement...")

        total_time = time() - start_time
        epochs_completed = epoch + 1

        print(f"\nTotal Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Epochs completed: {epochs_completed}/{NUM_EPOCHS}")
        print(f"Best test accuracy: {best_test_acc:.4f}")
        print(f"Best F1 score: {best_f1:.4f}")

        # Load best model and evaluate
        print("\n" + "=" * 70)
        print("Loading best model for final evaluation...")
        print("=" * 70)
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))

        # Run final evaluation on best model
        acc, prec, rec, f1, cm = evaluate(model, val_loader, device)

        print("\nBest Model Performance:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        # Per-class metrics
        print("\nPer-Class Performance:")
        for i, name in enumerate(CLASS_NAMES):
            class_total = cm[i].sum()
            class_correct = cm[i, i]
            class_acc = class_correct / class_total if class_total > 0 else 0
            print(f"  {name}: {class_correct}/{class_total} ({100*class_acc:.1f}%)")

        plot_confusion_matrix(cm, CLASS_NAMES)

        print("\nConfusion matrix saved to: confusion_matrix.png")
        print("=" * 70)

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print("Training log saved to: training_log.txt")


if __name__ == "__main__":
    main()
