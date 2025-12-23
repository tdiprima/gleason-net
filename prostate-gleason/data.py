"""
Dataset and data loading for Prostate Cancer Gleason Grading
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


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
            name_without_ext = filename.rsplit(".", 1)[0]
            # Try underscore first, then dash
            if "_" in name_without_ext:
                label_str = name_without_ext.rsplit("_", 1)[-1]
            else:
                label_str = name_without_ext.rsplit("-", 1)[-1]
            return int(label_str)
        except (ValueError, IndexError):
            return None

    def _load_samples(self):
        """Scan directory and load valid samples."""
        print(f"\nScanning dataset directory: {self.root_dir}")

        # Support both flat directory and nested structure
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
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

        except Exception:
            # Handle corrupted images by returning a valid placeholder
            # This prevents DataLoader from crashing
            if img_path not in self.corrupted_files:
                self.corrupted_files.append(img_path)
                print(f"Warning: Corrupted image skipped: {img_path}")

            # Return a black image with correct label
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


def get_loaders(data_root, batch_size, input_size, valid_labels, label_map,
                train_split, random_seed, num_workers, pin_memory, prefetch_factor):
    """
    Create train and validation data loaders.

    Returns:
        train_loader, val_loader
    """
    # Load full dataset with training transforms initially
    full_dataset = ProstatePathologyDataset(
        root_dir=data_root,
        transform=get_train_transforms(input_size),
        valid_labels=valid_labels,
        label_map=label_map,
    )

    if len(full_dataset) == 0:
        print("ERROR: No valid samples found in dataset!")
        print(f"Please check that {data_root} exists and contains properly named images.")
        return None, None

    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset_temp = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # Create a separate dataset for validation with proper transforms
    val_dataset_proper = ProstatePathologyDataset(
        root_dir=data_root,
        transform=get_val_transforms(input_size),
        valid_labels=valid_labels,
        label_map=label_map,
    )

    # Get the actual random indices from the split (not sequential!)
    val_indices = val_dataset_temp.indices
    val_dataset_proper = torch.utils.data.Subset(val_dataset_proper, val_indices)

    print("\nDataset Split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset_proper)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=custom_collate_fn,
        drop_last=True,  # Drop incomplete batches for stable batch norm
    )

    val_loader = DataLoader(
        val_dataset_proper,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=custom_collate_fn,
    )

    return train_loader, val_loader
