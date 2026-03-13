"""
Mask-only ROI-cropped gesture classifier.

Takes a single-channel cropped hand mask tensor of shape (B, 1, H, W)
and predicts gesture logits of shape (B, num_classes).

Used for the lightweight laptop version of the pipeline where no RGB or
depth is required at inference time -- only the segmenter mask.

The idea is that the shape of the hand silhouette alone should be enough
to distinguish gestures like "thumbs up" from "peace sign" etc. This is
kind of cool because it means the classifier is completely invariant to
lighting, skin tone, background etc. -- it only sees the blob shape.
I wasn't sure if this would work well but the results are pretty decent.
"""

import torch
import torch.nn as nn


class MaskOnlyEncoder(nn.Module):
    """
    Lightweight CNN encoder for mask-only input (1 channel: M).

    Same architecture pattern as the RGB encoder (Conv+BN+ReLU+MaxPool x4)
    but takes only 1 input channel instead of 3 or 4. The input is expected
    to be a cropped and resized binary mask of the hand region (224x224).

    After 4 MaxPool2d(2) layers, a 224x224 input becomes 14x14, which gets
    pooled down to 1x1 by the AdaptiveAvgPool in the classifier head.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 64 channels, 224 -> 112
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64 -> 128, 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 128 -> 256, 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 256 -> 512, 28 -> 14
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # just runs straight through the sequential, no skip connections
        return self.features(x)  # (B, 512, H/16, W/16)


class MROICropClassifier(nn.Module):
    """
    Mask-only ROI-cropped gesture classifier.

    Full pipeline:
        1. Segmenter produces a binary mask for the whole image
        2. We find the bounding box of the hand (largest connected component)
        3. Crop and resize to 224x224 -- just the mask channel, no colour info
        4. Feed into this classifier

    The crop-and-resize step is done before calling this model, in the training loop.
    This module only does the actual classification from the (B, 1, 224, 224) crop.

    Args:
        num_classes: Number of gesture classes (10 for this dataset)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = MaskOnlyEncoder()
        # Global average pooling to collapse spatial dims to 1x1
        # This makes the model input-size agnostic (though we always use 224x224)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Linear classifier head: 512 features -> num_classes logits
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode the mask crop, pool it, classify it.

        Args:
            x: (B, 1, H, W) tensor -- cropped binary hand mask, values in [0, 1]
               In practice H=W=224 but technically any size should work.

        Returns:
            Logits of shape (B, num_classes) -- NOT softmaxed, use CrossEntropyLoss directly.
        """
        feats = self.backbone(x)      # (B, 512, H/16, W/16)
        pooled = self.pool(feats).view(x.size(0), -1)  # (B, 512) -- flatten after pool
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


if __name__ == "__main__":
    # Quick test to make sure shapes come out right
    model = MROICropClassifier(num_classes=10)
    dummy = torch.randn(4, 1, 224, 224)  # batch of 4 mask crops
    out = model(dummy)
    print("Logits shape:", out.shape)  # should be (4, 10)
