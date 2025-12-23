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

from config import (BATCH_SIZE, CLASS_NAMES, DATA_ROOT, DEVICE, INPUT_SIZE,
                   LABEL_MAP, LEARNING_RATE, NON_BLOCKING, NUM_CLASSES,
                   NUM_EPOCHS, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR,
                   PRETRAINED, RANDOM_SEED, TRAIN_SPLIT, VALID_LABELS,
                   WEIGHT_DECAY)
from data import get_loaders
from eval import evaluate
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
            print(f"  Batch {batch_idx + 1}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def plot_confusion_matrix(cm, class_names):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Prostate Gleason Grading")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


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

        # Load data
        print("\n" + "=" * 70)
        print("LOADING DATASET")
        print("=" * 70)

        train_loader, val_loader = get_loaders(
            DATA_ROOT, BATCH_SIZE, INPUT_SIZE, VALID_LABELS, LABEL_MAP,
            TRAIN_SPLIT, RANDOM_SEED, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR
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

        # Loss function
        criterion = nn.CrossEntropyLoss()

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

        start_time = time()
        best_test_acc = 0.0

        for epoch in range(NUM_EPOCHS):
            epoch_start = time()
            current_lr = optimizer.param_groups[0]['lr']

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

            # Save best model
            if acc > best_test_acc:
                best_test_acc = acc
                torch.save(model.state_dict(), "best_model.pth")
                print("New best model (by test accuracy)")
            else:
                print("No improvement...")

        total_time = time() - start_time

        print(f"\nTotal Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Best test accuracy: {best_test_acc:.4f}")

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
        plot_confusion_matrix(cm, CLASS_NAMES)

        print("\nConfusion matrix saved to: confusion_matrix.png")
        print("=" * 70)

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print(f"Training log saved to: training_log.txt")


if __name__ == "__main__":
    main()
