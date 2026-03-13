"""
Bounding box analysis script for the segmentation-only HandAnalysisModel.

This script evaluates how well bounding boxes derived from the segmentation
mask match the ground-truth bounding boxes in the test dataset, and measures
the effect of a post-processing step that cleans up the predicted mask.

Pipeline for each test image:
  1. Run the segmenter to get a raw probability mask.
  2. Threshold at 0.5 to get a binary mask.
  3. Derive a tight bbox from that binary mask ("BEFORE" post-processing).
  4. Apply the largest connected component (LCC) filter to the binary mask
     to keep only the biggest foreground blob.
  5. Derive a tight bbox from the cleaned-up mask ("AFTER" post-processing).
  6. Compare both bboxes against the ground-truth bbox using IoU.

Metrics reported:
  - Mean bbox IoU before and after LCC post-processing
  - Detection accuracy @ IoU >= 0.5 before and after

We also save a few qualitative visualisations showing all three boxes
(GT in blue, before in red, after in green) on the actual image.

Usage (from project root):
    python src/boundingboxes.py --checkpoint weights/no_leaks_best.pt --data test_dataset --out results/no_leaks_bbox
"""

import argparse
import sys
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import ndimage
from PIL import Image as PILImage, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import GestureDataset  # noqa: E402
from src.model_rgbd_segmenter import RGBDSegmenter  # noqa: E402
from src.utils import (  # noqa: E402
    collate_fn,
    bbox_xyxy_norm_to_pixels,
    compute_bbox_iou,
)


def parse_args():
    p = argparse.ArgumentParser(description="Bounding box evaluation from segmentation masks")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to segmenter checkpoint (.pt)")
    p.add_argument("--data",       type=str, required=True,
                   help="Path to dataset root (e.g. test_dataset)")
    p.add_argument("--out",        type=str, required=True,
                   help="Output directory for metrics and visualisations (e.g. results/no_leaks_bbox)")
    p.add_argument("--batch_size", type=int,   default=8,
                   help="Batch size for evaluation")
    p.add_argument("--iou_thresh", type=float, default=0.5,
                   help="IoU threshold for detection accuracy (standard is 0.5)")
    p.add_argument("--vis_samples", type=int,  default=8,
                   help="Number of qualitative visualisation images to save")
    return p.parse_args()


# =========================================================================
# Post-processing
# =========================================================================

def largest_component_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask, discard all others.

    This is the main post-processing step. The segmenter sometimes predicts
    multiple scattered blobs of foreground pixels -- typically one large blob
    (the actual hand) and a few small noisy ones. This function keeps only
    the largest blob and zeros out everything else.

    The underlying idea is simple: the hand is always the dominant foreground
    region, so the biggest connected blob is almost certainly the hand.

    Args:
        binary_mask: (H, W) numpy array with values in {0, 1}

    Returns:
        (H, W) uint8 array with only the largest component kept.
        If the mask is completely empty, returns it unchanged.
    """
    if binary_mask.sum() == 0:
        return binary_mask  # nothing to clean up

    # Label connected components -- each separate blob gets a unique integer label
    labeled, num_features = ndimage.label(binary_mask)

    if num_features <= 1:
        return binary_mask  # already one blob (or empty), nothing to do

    # Count the number of foreground pixels in each component
    # index=range(1, n+1) skips the background (label 0)
    sizes         = ndimage.sum(binary_mask, labeled, index=range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1  # +1 because labels start at 1

    # Return a mask that is 1 only where the largest component was
    return (labeled == largest_label).astype(np.uint8)


# =========================================================================
# Bbox utilities
# =========================================================================

def mask_to_bbox_pixels(mask_binary: np.ndarray):
    """
    Find the tight axis-aligned bounding box of all foreground pixels in a binary mask.

    Args:
        mask_binary: (H, W) numpy array in {0, 1}

    Returns:
        (x1, y1, x2, y2) tuple of ints in pixel coordinates,
        or None if there are no foreground pixels.
    """
    ys, xs = np.where(mask_binary > 0)
    if ys.size == 0 or xs.size == 0:
        return None  # empty mask
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return (x1, y1, x2, y2)


# =========================================================================
# Visualisation
# =========================================================================

def draw_bboxes_on_rgb(rgb_tensor, gt_bbox, pred_before, pred_after, save_path):
    """
    Draw three bounding boxes on a sample image and save it.

    Colour coding:
        Blue  = ground truth bounding box from dataset annotations
        Red   = predicted bbox BEFORE LCC post-processing (from raw mask)
        Green = predicted bbox AFTER  LCC post-processing (from cleaned mask)

    This makes it easy to see visually whether the post-processing actually
    helped for each individual sample.

    Args:
        rgb_tensor:  (4, H, W) RGBD tensor -- we only use the first 3 channels
        gt_bbox:     (x1, y1, x2, y2) ground truth box in pixels
        pred_before: (x1, y1, x2, y2) predicted box before post-processing (or None)
        pred_after:  (x1, y1, x2, y2) predicted box after post-processing (or None)
        save_path:   Full path to save the output PNG image
    """
    # Extract RGB channels and convert to uint8 numpy array
    rgb_np = rgb_tensor[:3].cpu().numpy()  # (3, H, W)
    rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
    rgb_np = np.transpose(rgb_np, (1, 2, 0))  # (H, W, 3)

    img  = PILImage.fromarray(rgb_np).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw GT box in blue (underneath so it shows when overlapping)
    if gt_bbox is not None:
        draw.rectangle(gt_bbox, outline=(0, 0, 255), width=3)
    # Draw "before" box in red
    if pred_before is not None:
        draw.rectangle(pred_before, outline=(255, 0, 0), width=3)
    # Draw "after" box in green (on top -- this is what we care about most)
    if pred_after is not None:
        draw.rectangle(pred_after, outline=(0, 255, 0), width=3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    # Resolve all paths -- support both absolute and relative-to-project-root
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / args.data
    if not data_root.exists():
        print(f"ERROR: Data path does not exist: {data_root}")
        sys.exit(1)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset -- no augmentation for evaluation
    dataset = GestureDataset(str(data_root), transform=None)
    if len(dataset) == 0:
        print("ERROR: No samples in dataset.")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Load segmenter
    model = RGBDSegmenter().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Metric accumulators
    ious_before  = []  # per-sample bbox IoU before post-processing
    ious_after   = []  # per-sample bbox IoU after post-processing
    correct_before = 0  # count of correct detections @ iou_thresh before
    correct_after  = 0  # count of correct detections @ iou_thresh after
    total = 0

    # Visualisation state
    vis_saved = 0
    vis_limit = max(0, int(args.vis_samples))

    print(f"Evaluating bounding boxes on {len(dataset)} samples...")

    with torch.no_grad():
        for batch in loader:
            inp          = batch["input"].to(device)  # (B, 4, H, W)
            bbox_gt_norm = batch["bbox"]               # (B, 4) normalised XYXY
            B, _, H, W  = batch["mask"].shape

            # Forward pass -- get raw segmentation mask probabilities
            pred_mask = model(inp)  # (B, 1, H', W')

            # Resize prediction to match ground truth spatial size if needed
            if pred_mask.shape[2] != H or pred_mask.shape[3] != W:
                pred_mask = F.interpolate(pred_mask, size=(H, W), mode="bilinear", align_corners=False)

            for i in range(B):
                # Convert GT normalised bbox to pixel coordinates for comparison
                gt_bbox_pixels = bbox_xyxy_norm_to_pixels(bbox_gt_norm[i], height=H, width=W)

                # ---- BEFORE post-processing ----
                # Just threshold the raw mask at 0.5 and find tight bbox
                pm = (pred_mask[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
                pred_bbox_before = mask_to_bbox_pixels(pm)
                if pred_bbox_before is None:
                    iou_b = 0.0  # no prediction = IoU of 0
                else:
                    iou_b = compute_bbox_iou(pred_bbox_before, gt_bbox_pixels)

                # ---- AFTER post-processing (largest connected component) ----
                # Apply LCC filtering to remove noise blobs, then re-derive bbox
                pm_lcc = largest_component_mask(pm)
                pred_bbox_after = mask_to_bbox_pixels(pm_lcc)
                if pred_bbox_after is None:
                    iou_a = 0.0
                else:
                    iou_a = compute_bbox_iou(pred_bbox_after, gt_bbox_pixels)

                ious_before.append(iou_b)
                ious_after.append(iou_a)
                if iou_b >= args.iou_thresh:
                    correct_before += 1
                if iou_a >= args.iou_thresh:
                    correct_after += 1
                total += 1

                # Save qualitative visualisation for the first vis_limit samples
                if vis_saved < vis_limit:
                    rgb_tensor = batch["input"][i]  # keep on CPU
                    vis_path   = out_dir / "vis" / f"sample_{total:05d}.png"
                    draw_bboxes_on_rgb(
                        rgb_tensor,
                        gt_bbox_pixels,
                        pred_bbox_before,
                        pred_bbox_after,
                        str(vis_path),
                    )
                    vis_saved += 1

    # Compute final aggregate metrics
    mean_iou_before = float(np.mean(ious_before)) if ious_before else 0.0
    mean_iou_after  = float(np.mean(ious_after))  if ious_after  else 0.0
    acc_before = float(correct_before) / float(total) if total > 0 else 0.0
    acc_after  = float(correct_after)  / float(total) if total > 0 else 0.0

    metrics = {
        "num_samples":             int(total),
        "iou_threshold":           float(args.iou_thresh),
        "mean_bbox_iou_before":    mean_iou_before,
        "mean_bbox_iou_after":     mean_iou_after,
        "detection_accuracy_before": acc_before,
        "detection_accuracy_after":  acc_after,
    }

    # Save metrics to JSON
    with open(out_dir / "bbox_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Bounding box evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Saved metrics to {out_dir / 'bbox_eval_metrics.json'}")
    if vis_saved > 0:
        print(f"Saved {vis_saved} visualisation images to {out_dir / 'vis'}/")


if __name__ == "__main__":
    main()
