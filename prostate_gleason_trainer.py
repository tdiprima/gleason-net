"""
Prostate Cancer Gleason Grading Trainer
=======================================
ResNet50-based classification for prostate pathology patches.
4-way classification: Benign, Gleason 3, Gleason 4, Gleason 5

Dataset labels in filenames: *-{label}.png
- 0: Benign      -> Class 0
- 1: Gleason 3   -> Class 1
- 2: Gleason 4   -> Class 2
- 3: Gleason 5   -> Class 3
- 4, 5: Skipped (alternate Gleason 5 subtypes)

Patches: 250x250 pixels at 20X magnification
"""

import os
import warnings
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Training configuration parameters."""

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


# =============================================================================
# SYSTEM INFORMATION
# =============================================================================


def print_system_info():
    """Display system and PyTorch configuration."""
    print("=" * 70)
    print("PROSTATE CANCER GLEASON GRADING - RESNET50 TRAINER")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU - Training will be slow!")

    print(f"Training Device: {device}")
    print("=" * 70)
    return device


# =============================================================================
# DATASET
# =============================================================================


class ProstatePathologyDataset(Dataset):
    """
    Custom dataset for prostate pathology patches.
    
    Expects filenames with format: *-{label}.png
    Only loads patches with labels in VALID_LABELS (0, 1, 2, 3).
    Skips corrupted images gracefully.
    """

    def __init__(self, root_dir, transform=None, valid_labels=None, label_map=None):
        """
        Args:
            root_dir: Path to folder containing patch images
            transform: Optional transforms to apply
            valid_labels: Set of valid label values to include
            label_map: Dict mapping original labels to training labels
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.valid_labels = valid_labels or {0, 1, 2, 3}
        self.label_map = label_map or {0: 0, 1: 1, 2: 2, 3: 3}
        
        self.samples = []
        self.skipped_labels = {4: 0, 5: 0}
        self.corrupted_files = []
        
        self._load_samples()

    def _extract_label_from_filename(self, filename):
        """Extract label from filename pattern *_{label}.png or *-{label}.png"""
        try:
            # Get the part before .png and after the last dash or underscore
            name_without_ext = filename.rsplit('.', 1)[0]
            # Try underscore first, then dash
            if '_' in name_without_ext:
                label_str = name_without_ext.rsplit('_', 1)[-1]
            else:
                label_str = name_without_ext.rsplit('-', 1)[-1]
            return int(label_str)
        except (ValueError, IndexError):
            return None

    def _load_samples(self):
        """Scan directory and load valid samples."""
        print(f"\nScanning dataset directory: {self.root_dir}")
        
        # Support both flat directory and nested structure
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.root_dir.rglob(f"*{ext}"))
            image_files.extend(self.root_dir.rglob(f"*{ext.upper()}"))
        
        for img_path in image_files:
            label = self._extract_label_from_filename(img_path.name)
            
            if label is None:
                continue
            
            # Track skipped labels
            if label in {4, 5}:
                self.skipped_labels[label] += 1
                continue
            
            # Only include valid labels
            if label in self.valid_labels:
                self.samples.append((str(img_path), self.label_map[label]))
        
        print(f"Loaded {len(self.samples)} valid samples")
        print(f"Skipped label 4 (Gleason 5-Secretions): {self.skipped_labels[4]}")
        print(f"Skipped label 5 (Gleason 5): {self.skipped_labels[5]}")

    def _verify_image(self, img_path):
        """Verify image can be opened and is not corrupted."""
        try:
            with Image.open(img_path) as img:
                img.verify()
            # Re-open after verify (verify() leaves file in unusable state)
            with Image.open(img_path) as img:
                img.load()
            return True
        except Exception:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Open and convert to RGB (pathology images may be various formats)
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            # Handle corrupted images by returning a valid placeholder
            # This prevents DataLoader from crashing
            if img_path not in self.corrupted_files:
                self.corrupted_files.append(img_path)
                print(f"Warning: Corrupted image skipped: {img_path}")
            
            # Return a black image with correct label
            # The collate function will handle this
            if self.transform:
                placeholder = Image.new("RGB", (250, 250), (0, 0, 0))
                placeholder = self.transform(placeholder)
                return placeholder, label
            else:
                return Image.new("RGB", (250, 250), (0, 0, 0)), label


def custom_collate_fn(batch):
    """Custom collate function to filter out None values from corrupted images."""
    batch = [(img, label) for img, label in batch if img is not None]
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


# =============================================================================
# DATA TRANSFORMS
# =============================================================================


def get_train_transforms(input_size):
    """Training transforms with augmentation for pathology images."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(input_size):
    """Validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =============================================================================
# MODEL DEFINITION
# =============================================================================


def create_resnet50_model(num_classes, pretrained=True):
    """
    Create ResNet50 model with custom classification head.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        Modified ResNet50 model
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        print("Loaded ImageNet pretrained weights (ResNet50_Weights.IMAGENET1K_V2)")
    else:
        model = models.resnet50(weights=None)
        print("Training from scratch (no pretrained weights)")
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, 
                non_blocking=True, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    num_batches = len(train_loader)
    log_interval = max(1, num_batches // 5)  # Log ~5 times per epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        if data is None:
            continue
            
        # Transfer data to device
        data = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Progress logging
        if (batch_idx + 1) % log_interval == 0:
            current_acc = 100.0 * correct / total
            print(f"  Batch {batch_idx + 1}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")

    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, device, data_loader, criterion, non_blocking=True, 
             class_names=None):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class metrics
    num_classes = len(class_names) if class_names else 4
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for data, target in data_loader:
            if data is None:
                continue
                
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            output = model(data)
            total_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Per-class accuracy
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if pred[i] == label:
                    class_correct[label] += 1

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {}
    if class_names:
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                class_accuracies[name] = 100.0 * class_correct[i] / class_total[i]
            else:
                class_accuracies[name] = 0.0

    return avg_loss, accuracy, class_accuracies


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def main():
    """Main training function."""
    config = Config()

    # Print system information and get device
    device = print_system_info()

    # Adjust for CPU if needed (though we expect GPU)
    if not torch.cuda.is_available():
        config.NUM_WORKERS = 4
        config.PIN_MEMORY = False
        config.NON_BLOCKING = False
        print("\nWARNING: GPU not available, adjusting settings for CPU")

    # Print configuration
    print("\nConfiguration:")
    print(f"  Data Root: {config.DATA_ROOT}")
    print(f"  Number of Classes: {config.NUM_CLASSES}")
    print(f"  Input Size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Weight Decay: {config.WEIGHT_DECAY}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Workers: {config.NUM_WORKERS}")
    print(f"  Pin Memory: {config.PIN_MEMORY}")
    print(f"  Pretrained: {config.PRETRAINED}")
    print(f"  Valid Labels: {config.VALID_LABELS}")

    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)

    # Create datasets
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    # Load full dataset with training transforms initially
    full_dataset = ProstatePathologyDataset(
        root_dir=config.DATA_ROOT,
        transform=get_train_transforms(config.INPUT_SIZE),
        valid_labels=config.VALID_LABELS,
        label_map=config.LABEL_MAP
    )
    
    if len(full_dataset) == 0:
        print("ERROR: No valid samples found in dataset!")
        print(f"Please check that {config.DATA_ROOT} exists and contains properly named images.")
        return None

    # Split into train and validation
    train_size = int(config.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Note: For proper validation, we should use val transforms on val set
    # Create a separate dataset for validation with proper transforms
    val_dataset_proper = ProstatePathologyDataset(
        root_dir=config.DATA_ROOT,
        transform=get_val_transforms(config.INPUT_SIZE),
        valid_labels=config.VALID_LABELS,
        label_map=config.LABEL_MAP
    )
    
    # Get the same indices for validation
    val_indices = val_dataset.indices
    val_dataset_proper = torch.utils.data.Subset(val_dataset_proper, val_indices)

    print(f"\nDataset Split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset_proper)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
        collate_fn=custom_collate_fn,
        drop_last=True  # Drop incomplete batches for stable batch norm
    )

    val_loader = DataLoader(
        val_dataset_proper,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
        collate_fn=custom_collate_fn
    )

    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    model = create_resnet50_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED
    )
    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Loss function with class weights (optional, uncomment if needed for imbalanced data)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    start_time = time()
    best_val_accuracy = 0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} (LR: {current_lr:.6f})")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch,
            non_blocking=config.NON_BLOCKING
        )

        # Validate
        val_loss, val_acc, class_accs = evaluate(
            model, device, val_loader, criterion,
            non_blocking=config.NON_BLOCKING,
            class_names=config.CLASS_NAMES
        )
        
        # Step scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time() - epoch_start
        
        # Print epoch summary
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Per-class Val Accuracy:")
        for name, acc in class_accs.items():
            print(f"    {name}: {acc:.2f}%")
        print(f"  Epoch Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'config': {
                    'num_classes': config.NUM_CLASSES,
                    'input_size': config.INPUT_SIZE,
                    'class_names': config.CLASS_NAMES
                }
            }, 'best_prostate_resnet50.pth')
            print(f"  *** New best model saved (Val Acc: {val_acc:.2f}%) ***")

    total_time = time() - start_time

    # Final results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average Time per Epoch: {total_time / config.NUM_EPOCHS:.1f} seconds")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% (Epoch {best_epoch})")

    # Load best model for final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (Best Model)")
    print("=" * 70)
    
    checkpoint = torch.load('best_prostate_resnet50.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_loss, final_acc, final_class_accs = evaluate(
        model, device, val_loader, criterion,
        non_blocking=config.NON_BLOCKING,
        class_names=config.CLASS_NAMES
    )
    
    print(f"Final Validation Loss: {final_loss:.4f}")
    print(f"Final Validation Accuracy: {final_acc:.2f}%")
    print(f"\nPer-class Accuracy:")
    for name, acc in final_class_accs.items():
        print(f"  {name}: {acc:.2f}%")
    print("=" * 70)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_classes': config.NUM_CLASSES,
            'input_size': config.INPUT_SIZE,
            'class_names': config.CLASS_NAMES,
            'label_map': config.LABEL_MAP
        },
        'final_accuracy': final_acc,
        'history': history
    }, 'final_prostate_resnet50.pth')
    print("\nFinal model saved to: final_prostate_resnet50.pth")
    print("Best model saved to: best_prostate_resnet50.pth")

    return model, history


if __name__ == "__main__":
    result = main()
    if result is not None:
        model, history = result
        print(model)
        print(history)
