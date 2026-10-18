"""
CNN Feature Extractor for dimensionality reduction.

Reduces high-dimensional images (e.g., MNIST 28x28=784) to low-dimensional
feature maps suitable for spike encoding and MEA stimulation.

Reference: Wang et al. (2025) - 3 conv layers + 3 max pooling layers,
producing 8 feature maps of 5x5 = 200 values from color images.

For MNIST (28x28 grayscale), we adapt the architecture to produce
N_channels feature maps of spatial_size x spatial_size, where N_channels
matches the number of stimulation electrodes (default: 8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """
    CNN that compresses images into small feature maps for spike encoding.

    Architecture follows Wang et al. but adapted for MNIST:
        Conv(1, 12, 5) -> ReLU -> MaxPool(2)  : 28->12 (with padding)
        Conv(12, 14, 5) -> ReLU -> MaxPool(2)  : 12->4
        Conv(14, n_channels, 3) -> ReLU         : 4->4 (keep spatial dims)

    Output: (batch, n_channels, spatial_h, spatial_w)
    Each channel maps to one stimulation electrode.
    """

    def __init__(self, n_channels=8, n_classes=10, input_size=28):
        """
        Args:
            n_channels: Number of output feature map channels.
                        Should match number of stimulation electrodes.
            n_classes: Number of classification targets (for training).
            input_size: Input image spatial size (28 for MNIST).
        """
        super().__init__()
        self.n_channels = n_channels

        # Feature extraction layers
        # MNIST 28x28 -> Conv(5,pad=2) -> 28x28 -> Pool(2) -> 14x14
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=2)
        # 14x14 -> Conv(5,pad=2) -> 14x14 -> Pool(2) -> 7x7
        self.conv2 = nn.Conv2d(12, 14, kernel_size=5, stride=1, padding=2)
        # 7x7 -> Conv(3,pad=1) -> 7x7 -> Pool(2) -> 3x3 (floor)
        self.conv3 = nn.Conv2d(14, n_channels, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Spatial size after 3 pooling layers: floor(28/2/2/2) = 3
        self.spatial_size = input_size // 8  # = 3 for MNIST

        # Classification head (for supervised training of the feature extractor)
        self.classifier = nn.Linear(
            n_channels * self.spatial_size * self.spatial_size, n_classes
        )

    def extract_features(self, x):
        """
        Extract normalized feature maps without classification.

        Args:
            x: (batch, 1, 28, 28) input images, values in [0, 1].

        Returns:
            features: (batch, n_channels, spatial_h, spatial_w),
                      values normalized to [0, 1] per sample.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Normalize to [0, 1] per sample for encoding
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        x_min = x_flat.min(dim=1, keepdim=True)[0]
        x_max = x_flat.max(dim=1, keepdim=True)[0]
        x_range = (x_max - x_min).clamp(min=1e-8)
        x_flat = (x_flat - x_min) / x_range
        x = x_flat.view_as(x)

        return x

    def forward(self, x):
        """Full forward pass including classification (for training)."""
        features = self.extract_features(x)
        flat = features.view(features.size(0), -1)
        logits = self.classifier(flat)
        return logits

    def get_feature_map_size(self):
        """Return (n_channels, spatial_h, spatial_w)."""
        return (self.n_channels, self.spatial_size, self.spatial_size)


def train_feature_extractor(
    train_loader,
    val_loader,
    n_channels=8,
    n_classes=10,
    n_epochs=50,
    lr=5e-4,
    device="cpu",
    early_stopping_patience=5,
):
    """
    Train the CNN feature extractor on a classification task.

    After training, the classifier head is discarded and only the
    feature extraction layers are used.

    Args:
        train_loader: DataLoader with (images, labels).
        val_loader: DataLoader for validation.
        n_channels: Number of feature map channels.
        n_classes: Number of classes.
        n_epochs: Maximum training epochs.
        lr: Learning rate (Wang et al. used 5e-4 with Adam).
        device: "cpu" or "cuda".
        early_stopping_patience: Stop if val loss doesn't improve for N epochs.

    Returns:
        model: Trained CNNFeatureExtractor.
        history: Dict with training/validation loss and accuracy per epoch.
    """
    model = CNNFeatureExtractor(n_channels=n_channels, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        history["train_loss"].append(total_loss / total)
        history["train_acc"].append(correct / total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        epoch_val_loss = val_loss / val_total
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_correct / val_total)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
