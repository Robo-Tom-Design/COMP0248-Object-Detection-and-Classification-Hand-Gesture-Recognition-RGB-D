"""
Utility functions for the Hand Analysis model.

This file is basically a grab-bag of helper functions that get imported by
multiple other scripts. Things like the collate function, metric computations,
bbox coordinate conversion, and a running average meter.
"""

import torch
import numpy as np
from PIL import Image
from scipy import ndimage


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    PyTorch's default collate_fn tries to stack everything into tensors, which
    works fine for our case -- but we define our own so we have explicit control
    over the output dict keys and can easily add new fields later.

    Args:
        batch: List of dicts from GestureDataset.__getitem__

    Returns:
        Dict with stacked tensors:
            'input':   (B, 4, H, W) float -- RGBD model inputs
            'mask':    (B, 1, H, W) float -- ground truth segmentation masks
            'bbox':    (B, 4) float       -- normalised XYXY bounding boxes
            'gesture': (B,) long          -- gesture class labels
    """
    inputs = torch.stack([s['input'] for s in batch], dim=0)
    masks = torch.stack([s['mask'] for s in batch], dim=0)
    bboxes = torch.stack([s['bbox'] for s in batch], dim=0)
    gestures = torch.stack([s['gesture'] for s in batch], dim=0)
    return {
        'input': inputs,
        'mask': masks,
        'bbox': bboxes,
        'gesture': gestures,
    }


def process_mask(mask, threshold=0.5):
    """
    Post-process a predicted segmentation mask.

    Binarises the mask at the given threshold and removes small connected
    components (noise blobs). This is a simpler version of the LCC filtering
    -- it just throws away any component with fewer than 100 pixels, rather
    than specifically keeping the largest one.

    Args:
        mask:      (H, W) numpy array with predicted probabilities in [0, 1]
        threshold: Probability threshold for binarisation (default 0.5)

    Returns:
        Binary mask (H, W) as uint8 with small components removed.
    """
    binary_mask = (mask > threshold).astype(np.uint8)

    # Use scipy's connected component labelling to find separate blobs
    labeled, num_features = ndimage.label(binary_mask)
    # Count the size (number of pixels) of each component
    sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))

    # Zero out any component smaller than 100 pixels (probably noise)
    mask_size = sizes < 100
    binary_mask[mask_size[labeled]] = 0

    return binary_mask


def process_bbox(bbox, image_shape):
    """
    Post-process a predicted bounding box from normalised format to pixel coords.

    Note: this function expects the bbox in (x_centre, y_centre, width, height)
    format, not XYXY. This is a legacy format from an older version of the model.
    The current model/dataloader uses XYXY normalised format -- use
    bbox_xyxy_norm_to_pixels() instead for those.

    Args:
        bbox:        (4,) array/tensor with (x_norm, y_norm, w_norm, h_norm)
        image_shape: (height, width) of the original image in pixels

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, clamped to image bounds.
    """
    h, w = image_shape
    x_norm, y_norm, w_norm, h_norm = bbox

    # Denormalise centre coordinates and size
    x = int(x_norm * w)
    y = int(y_norm * h)
    width = int(w_norm * w)
    height = int(h_norm * h)

    # Convert from centre+size to corner format (XYXY)
    x1 = max(0, x - width // 2)
    y1 = max(0, y - height // 2)
    x2 = min(w, x1 + width)
    y2 = min(h, y1 + height)

    return (x1, y1, x2, y2)


def extract_hand_region(image, bbox):
    """
    Simple crop of the hand region from a full image using a bounding box.

    Args:
        image: (H, W, C) numpy array
        bbox:  (x1, y1, x2, y2) pixel coordinates

    Returns:
        Cropped numpy array of the hand region.
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU) between two binary masks.

    This is the standard segmentation metric. A value of 1.0 means perfect overlap,
    0.0 means no overlap at all. We use this during training to track progress.

    Args:
        pred_mask: (H, W) numpy array -- predicted binary mask
        gt_mask:   (H, W) numpy array -- ground truth binary mask

    Returns:
        IoU score as float in [0, 1].
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    # Handle edge case where both masks are empty -- convention is IoU = 0 for empty predictions
    if union == 0:
        return 0.0

    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """
    Compute the Dice coefficient between two binary masks.

    Dice = 2 * |intersection| / (|pred| + |gt|)

    Dice is related to IoU (Dice = 2*IoU / (1+IoU)) but tends to be more forgiving
    of small misalignments. We track both during training.

    Special case: if both masks are completely empty, we return 1.0 (perfect score)
    if they match exactly, or 0.0 otherwise. This handles the no-hand case.

    Args:
        pred_mask: (H, W) binary array
        gt_mask:   (H, W) binary array

    Returns:
        Dice coefficient as float in [0, 1].
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()

    # Handle case where both are empty
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 1.0 if np.array_equal(pred_mask, gt_mask) else 0.0

    return 2 * intersection / (pred_mask.sum() + gt_mask.sum())


def bbox_xyxy_norm_to_pixels(bbox_xyxy_norm, height, width):
    """
    Convert normalised XYXY bounding box coordinates to integer pixel coordinates.

    Our dataloader stores bboxes normalised to [0, 1] relative to image dimensions.
    This function converts them back to pixel coordinates for metric computation
    and visualisation.

    Args:
        bbox_xyxy_norm: (4,) or (N, 4) array/tensor -- x_min, y_min, x_max, y_max in [0,1]
        height:         Image height in pixels
        width:          Image width in pixels

    Returns:
        (x1, y1, x2, y2) as a tuple of ints (for single bbox),
        or a list of such tuples for N > 1.
    """
    if isinstance(bbox_xyxy_norm, torch.Tensor):
        bbox_xyxy_norm = bbox_xyxy_norm.cpu().numpy()
    b = np.asarray(bbox_xyxy_norm, dtype=np.float64)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Scale and clip to valid pixel range
    x1 = np.clip(b[:, 0] * width,  0, width  - 1).astype(int)
    y1 = np.clip(b[:, 1] * height, 0, height - 1).astype(int)
    x2 = np.clip(b[:, 2] * width,  0, width  - 1).astype(int)
    y2 = np.clip(b[:, 3] * height, 0, height - 1).astype(int)

    if len(x1) == 1:
        return (int(x1[0]), int(y1[0]), int(x2[0]), int(y2[0]))
    return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])) for i in range(len(x1))]


def compute_bbox_iou(pred_bbox, gt_bbox):
    """
    Compute IoU between two bounding boxes in pixel XYXY format.

    This is the standard detection IoU used for the "detection accuracy @ 0.5 IoU"
    metric. A detection is considered correct if its IoU with the ground truth
    box is >= 0.5 (or whatever threshold you're using).

    Args:
        pred_bbox: (x1, y1, x2, y2) predicted box in pixels
        gt_bbox:   (x1, y1, x2, y2) ground truth box in pixels

    Returns:
        IoU score as float in [0, 1].
    """
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox
    x1_gt,   y1_gt,   x2_gt,   y2_gt   = gt_bbox

    # Compute intersection rectangle by taking max of the left/top edges
    # and min of the right/bottom edges
    x1_i = max(x1_pred, x1_gt)
    y1_i = max(y1_pred, y1_gt)
    x2_i = min(x2_pred, x2_gt)
    y2_i = min(y2_pred, y2_gt)

    # If intersection is empty (boxes don't overlap), IoU is 0
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    inter_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Individual box areas
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area   = (x2_gt   - x1_gt)   * (y2_gt   - y1_gt)

    # Union = sum of areas minus the intersection (which was double-counted)
    union_area = pred_area + gt_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class AverageMeter:
    """
    Utility class for tracking running averages during training.

    Used to compute the mean loss, IoU, dice etc. over an epoch without
    having to store all individual values. Just call update() after each
    batch and read .avg at the end of the epoch.

    Example:
        meter = AverageMeter()
        for batch in loader:
            loss = compute_loss(...)
            meter.update(loss.item(), n=batch_size)
        print(f"Epoch mean loss: {meter.avg:.4f}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked values back to zero."""
        self.val   = 0  # most recent value
        self.avg   = 0  # running average
        self.sum   = 0  # running total (weighted by n)
        self.count = 0  # total number of samples seen

    def update(self, val, n=1):
        """
        Add a new observation.

        Args:
            val: The metric value (e.g. loss for this batch)
            n:   Number of samples this value represents (e.g. batch size)
                 Important for computing weighted averages correctly.
        """
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count > 0 else 0


def save_predictions(predictions, save_dir, prefix='pred'):
    """
    Save model predictions to disk as images and text files.

    This is a basic debug helper -- saves the mask as a PNG and the bbox/gesture
    as text files. Mainly useful during development to quickly inspect outputs.

    Args:
        predictions: Dict with 'mask' (H,W array), 'bbox' (4-tuple), 'gesture' (int)
        save_dir:    Directory path to save into
        prefix:      Filename prefix for all saved files
    """
    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Save mask as PNG -- scale from [0,1] float to [0,255] uint8
    mask = (predictions['mask'] * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask)
    mask_image.save(save_dir / f'{prefix}_mask.png')

    # Save bbox as a plain text file (not the most sophisticated output but functional)
    bbox_info = predictions['bbox']
    with open(save_dir / f'{prefix}_bbox.txt', 'w') as f:
        f.write(f"x1, y1, x2, y2: {bbox_info}\n")

    # Save predicted gesture class index
    with open(save_dir / f'{prefix}_gesture.txt', 'w') as f:
        f.write(f"Gesture class: {predictions['gesture']}\n")


if __name__ == '__main__':
    print("Testing utility functions...")

    # Test pixel-level IoU
    pred_mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    gt_mask   = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool)

    iou  = compute_iou(pred_mask, gt_mask)
    dice = compute_dice(pred_mask, gt_mask)

    print(f"IoU: {iou:.4f}")    # should be 4/6 = 0.6667
    print(f"Dice: {dice:.4f}")  # should be 2*4/(4+6) = 0.8

    # Test bounding box IoU
    pred_bbox = (10, 10, 50, 50)
    gt_bbox   = (15, 15, 55, 55)
    bbox_iou  = compute_bbox_iou(pred_bbox, gt_bbox)
    print(f"BBox IoU: {bbox_iou:.4f}")
