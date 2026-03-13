"""
RGB-D hand segmentation model.

This module defines the segmentation-only network that takes RGB-D input
and predicts a hand mask. It is the cleaned-up, final version of the
segmenter used throughout the project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBDEncoder(nn.Module):
    """
    Lightweight encoder backbone for RGB-D input (4 channels).

    This is basically a classic encoder-style CNN -- Conv+BN+ReLU+MaxPool blocks
    stacked up to progressively downsample the spatial dimensions while increasing
    the number of feature channels. Nothing fancy, but it works well enough.

    Input: (B, 4, H, W) -- 3 RGB channels + 1 normalised depth channel
    Output: (B, 512, H/32, W/32) -- heavily downsampled feature map

    I originally tried a UNet-style skip-connection architecture but the simpler
    encoder-decoder without skips was easier to debug and still got decent IoU,
    so I kept it like this.
    """

    def __init__(self):
        super().__init__()
        # Main feature extraction blocks -- each doubles the channels and halves spatial res
        # bias=False because we have BatchNorm after each conv, which has its own bias term
        self.features = nn.Sequential(
            # Block 1: 4 channels -> 64 channels, H/2 x W/2
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64 -> 128 channels, H/4 x W/4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 128 -> 256 channels, H/8 x W/8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 256 -> 512 channels, H/16 x W/16
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Extra block to get one more level of downsampling (H/32 x W/32)
        # kept it separate in case we want to optionally skip it later
        self.extra_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.extra_block(x)
        return x  # shape: (B, 512, H/32, W/32)


class RGBDSegmenter(nn.Module):
    """
    Segmentation-only model for hand analysis using RGB-D input.
    Predicts a pixel-wise hand mask from 4-channel RGB-D images.

    Architecture is encoder-decoder style:
        - Encoder: RGBDEncoder (see above) -- downsamples to small feature map
        - Decoder: series of transposed convolutions to upsample back to image size
        - Head: 1x1 conv to get single-channel mask logits, followed by sigmoid

    The decoder adds 4 upsampling steps (each ConvTranspose2d doubles spatial res)
    which brings us from H/32 back to H/2 -- note this means the output is still
    half the input resolution. We use F.interpolate in the loss/eval to resize up
    if needed (see SegmentationLoss forward).
    """

    def __init__(self):
        super().__init__()
        self.backbone = RGBDEncoder()
        backbone_out_channels = 512  # matches the last conv in RGBDEncoder

        # Decoder -- transposed convolutions upsample 2x each step
        # kernel_size=4, stride=2, padding=1 gives exact 2x upsampling
        self.seg_decoder = nn.Sequential(
            # 512 -> 256, x2 upsample
            nn.ConvTranspose2d(backbone_out_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 -> 128, x2 upsample
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 -> 64, x2 upsample
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 -> 32, x2 upsample -- now at H/2
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Final 1x1 conv to collapse channels to 1 (binary mask)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder then decoder.

        Args:
            x: (B, 4, H, W) RGB-D tensor -- depth channel should be normalised to [0,1]
               If running RGB-only (no depth sensor), just pass zeros for channel 3.

        Returns:
            Segmentation mask probabilities (B, 1, H', W') in [0, 1]
            where H' and W' are approximately H/2 and W/2 (half input resolution).
            The caller is responsible for upsampling to the full resolution if needed.
        """
        # Encode: spatial dims shrink, channels grow
        features = self.backbone(x)  # (B, 512, H/32, W/32)

        # Decode: spatial dims grow back, channels shrink
        seg_output = self.seg_decoder(features)  # (B, 32, H/2, W/2)

        # Apply sigmoid to convert raw logits to probabilities
        mask = torch.sigmoid(self.seg_head(seg_output))  # (B, 1, H/2, W/2) in [0,1]
        return mask


class SegmentationLoss(nn.Module):
    """
    Custom segmentation loss combining soft Dice and soft IoU.

    I chose this over plain binary cross-entropy because BCE treats every pixel
    independently and can struggle with class imbalance (background >> foreground).
    Dice and IoU losses are region-based and handle this better -- they directly
    optimise for overlap which is what we measure at eval time anyway.

    Both Dice and IoU loss are computed in their "soft" form (using probabilities
    rather than binary predictions), which means they're differentiable.
    """

    def __init__(self, use_dice: bool = True, use_iou: bool = True, eps: float = 1e-6):
        """
        Args:
            use_dice: Whether to include the Dice loss term. Default True.
            use_iou:  Whether to include the IoU loss term. Default True.
            eps:      Small constant for numerical stability in denominators.
                      Prevents division by zero when masks are empty.
        """
        super().__init__()
        if not (use_dice or use_iou):
            raise ValueError("At least one of use_dice or use_iou must be True.")
        self.use_dice = use_dice
        self.use_iou = use_iou
        self.eps = eps

    def forward(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined segmentation loss.

        Args:
            pred_mask:   Predicted probabilities (B, 1, H, W) in [0, 1]
            target_mask: Ground truth binary mask (B, 1, H, W) in {0, 1} (or soft [0,1])

        Returns:
            Scalar loss tensor (mean over batch).
        """
        # Clamp target just in case augmentation pushed any values outside [0,1]
        # CUDA BCELoss kernel asserts values are in range, so this is important
        target_mask = target_mask.clamp(0.0, 1.0)

        # If spatial sizes differ (e.g. model output is half-res), upsample pred to match target
        if pred_mask.shape != target_mask.shape:
            pred_mask = F.interpolate(
                pred_mask,
                size=target_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Flatten spatial dimensions so we can do dot products across all pixels at once
        # Shape goes from (B, 1, H, W) to (B, H*W)
        pred_flat = pred_mask.view(pred_mask.size(0), -1)
        target_flat = target_mask.view(target_mask.size(0), -1)

        # Compute shared statistics needed for both Dice and IoU
        intersection = (pred_flat * target_flat).sum(dim=1)  # (B,) -- element-wise product then sum
        pred_sum = pred_flat.sum(dim=1)    # total predicted foreground per image
        target_sum = target_flat.sum(dim=1)  # total gt foreground per image
        union = pred_sum + target_sum - intersection  # union = A + B - intersection

        losses = []

        if self.use_dice:
            # Soft Dice = 2*|intersection| / (|pred| + |target|)
            # Dice loss = 1 - Dice so that lower is better
            dice = (2 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
            dice_loss = 1.0 - dice  # (B,) -- one value per image in batch
            losses.append(dice_loss)

        if self.use_iou:
            # Soft IoU = |intersection| / |union|
            # IoU loss = 1 - IoU
            iou = (intersection + self.eps) / (union + self.eps)
            iou_loss = 1.0 - iou  # (B,)
            losses.append(iou_loss)

        # Sum the loss terms (so if both are used, total is up to 2.0 max)
        # then take the mean across the batch
        loss = sum(losses)
        loss = loss.mean()
        return loss


if __name__ == "__main__":
    # Quick sanity check -- run this file directly to make sure shapes are right
    model = RGBDSegmenter()
    x = torch.randn(4, 4, 480, 640)  # batch of 4 full-res RGB-D frames
    outputs = model(x)
    print("Model output mask shape:", outputs.shape)  # expect (4, 1, 240, 320) roughly

    loss_fn = SegmentationLoss(use_dice=True, use_iou=True)
    targets = torch.randint(0, 2, (4, 1, 480, 640)).float()
    loss = loss_fn(outputs, targets)
    print("Segmentation loss:", float(loss.item()))
