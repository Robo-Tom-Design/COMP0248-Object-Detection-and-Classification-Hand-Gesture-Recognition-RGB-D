"""
Full RGBD → Mask-only pipeline visualisation.

This script is hard-coded to:
  - Load the RGBD segmenter from `weights/overnight/best_rgbd_seg.pt`
  - Load the mask-only classifier from `weights/overnight/best_m_clf.pt`
  - Run on the `test_dataset` folder
  - Sample a bunch of test frames and, for each one, run:
        RGBD image → segmentation mask M → largest connected component
        → predicted hand bounding box → cropped mask → mask-only classifier

Each saved PNG shows:
  - RGB frame with predicted segmentation overlaid in transparent red
  - Ground-truth bounding box in blue
  - Predicted bounding box derived from M in green
  - GT and predicted gesture labels rendered as text at the top

Usage (from project root):
    python src/visualise.py

You do not need to pass any arguments; all paths are hard-coded to the
overnight weights and the `test_dataset` directory. Change the constants
below if you want to point at different files.
"""

import argparse
import random
import sys
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage, ImageDraw, ImageFont
from scipy import ndimage
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # SSH-safe backend -- no display needed, saves to file only
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration -- edit these to point at your data and weights
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = str(_PROJECT_ROOT / "test_dataset")  # path to test set
SEG_WEIGHTS_PATH = str(_PROJECT_ROOT / "weights" / "overnight" / "best_rgbd_seg.pt")
CLF_WEIGHTS_PATH = str(_PROJECT_ROOT / "weights" / "overnight" / "best_m_clf.pt")
OUTPUT_DIR = str(_PROJECT_ROOT / "results" / "visualise_rgbd_mask")  # where to save images
NUM_SAMPLES = 8  # how many random test images to visualise
RANDOM_SEED = 42  # for reproducible sampling

# Allow running from project root (python src/visualise.py) or from src/ (python visualise.py)
_project_root_str = str(_PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

from src.dataloader import GestureDataset, GESTURE_TO_CLASS, GESTURE_LABELS  # noqa: E402
from src.model_rgbd_segmenter import RGBDSegmenter  # noqa: E402
from src.model_m_roi_crop_classifier import MROICropClassifier  # noqa: E402
from src.utils import collate_fn  # noqa: E402


def overlay_mask_on_image(base_img, mask, color=(255, 0, 0), alpha=0.4):
    """
    Overlay a semitransparent coloured mask on top of an RGB image.

    The mask is expected to be a (H, W) float array in [0, 1].
    Pixels where mask > 0 will be tinted with the given colour at the given alpha.
    This is purely for visualisation -- it doesn't affect metrics.

    Args:
        base_img: (H, W, 3) uint8 numpy array
        mask:     (H, W) float array in [0, 1], values > 0 = foreground
        color:    RGB tuple for the overlay tint
        alpha:    Blend factor (0 = no overlay, 1 = full overlay colour)

    Returns:
        PIL RGBA image with the mask blended in.
    """
    mask = np.clip(mask, 0, 1)
    # Create a solid-colour image the same size as base, masked by foreground pixels
    mask_img = np.zeros_like(base_img, dtype=np.uint8)
    mask_img[..., 0] = color[0]
    mask_img[..., 1] = color[1]
    mask_img[..., 2] = color[2]
    mask_img = (mask_img * mask[..., None]).astype(np.uint8)  # zero out background pixels
    overlay  = PILImage.fromarray(mask_img).convert("RGBA")
    base_rgba = PILImage.fromarray(base_img).convert("RGBA")
    # PIL.blend does a linear interpolation: out = base*(1-alpha) + overlay*alpha
    blended  = PILImage.blend(base_rgba, overlay, alpha)
    return blended


def draw_bbox(draw, bbox, outline=(0,255,0), width=3):
    """
    Draw a bounding box on a PIL ImageDraw canvas.

    Args:
        draw:    PIL ImageDraw object
        bbox:    (x1, y1, x2, y2) pixel coordinates
        outline: RGB colour for the box
        width:   Line width in pixels
    """
    if bbox is None:
        return
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=outline, width=width)


def largest_component_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask, discard all others.
    """
    if binary_mask.sum() == 0:
        return binary_mask

    labeled, num_features = ndimage.label(binary_mask)
    if num_features <= 1:
        return binary_mask

    sizes = ndimage.sum(binary_mask, labeled, index=range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return (labeled == largest_label).astype(np.uint8)


def crop_mask_for_classifier(mask_lcc: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """
    Crop the LCC mask to its bbox and resize to (1, 1, target_size, target_size).
    """
    ys, xs = np.where(mask_lcc > 0)
    if ys.size == 0 or xs.size == 0:
        crop = np.zeros((target_size, target_size), dtype=np.float32)
        return torch.from_numpy(crop)[None, None, ...]

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    y2 = max(y2, y1 + 1)
    x2 = max(x2, x1 + 1)

    crop = mask_lcc[y1:y2, x1:x2].astype(np.float32)
    crop_t = torch.from_numpy(crop)[None, None, ...]
    crop_t = F.interpolate(crop_t, size=(target_size, target_size), mode="nearest")
    return crop_t


def main(
    dataset_dir,
    seg_weights_path=None,
    clf_weights_path=None,
    output_dir="visualise_outputs",
    num_samples=8,
    seed=42,
):
    """
    Main visualisation function.

    Loads the model and dataset, picks num_samples random test images,
    runs inference, and saves annotated output images.
    """
    # Set random seeds for reproducible sampling
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the test dataset (no transform -- we want the raw unaugmented images)
    val_dataset = GestureDataset(dataset_dir, transform=None)
    print(f"Total test samples: {len(val_dataset)}")
    if len(val_dataset) == 0:
        print("No samples found. Check DATASET_DIR points to a folder with student subdirs (e.g. 25050766_CHENG/...).")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load segmentation model and restore weights from checkpoint
    seg_model = RGBDSegmenter()
    if seg_weights_path is not None and os.path.exists(seg_weights_path) and seg_weights_path.endswith((".pth", ".pt")):
        checkpoint = torch.load(seg_weights_path, map_location=device)
        # Checkpoints saved by train.py are dicts with a model_state_dict key
        state = checkpoint.get("model_state_dict", checkpoint)
        try:
            seg_model.load_state_dict(state, strict=True)
            print(f"Loaded segmenter weights from {seg_weights_path}")
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                # This happens if the checkpoint was saved from a different architecture
                print(f"WARNING: Checkpoint is from a different model architecture. Using random weights.")
                print(f"  Use a checkpoint saved by this project's train.py.")
            else:
                raise
    else:
        if seg_weights_path and not os.path.exists(seg_weights_path):
            print(f"Segmenter weights file not found: {seg_weights_path}")
        elif seg_weights_path and not seg_weights_path.endswith((".pth", ".pt")):
            print(f"PyTorch expects .pth/.pt checkpoints; got {seg_weights_path}")
        print("Using randomly initialised segmenter.")

    seg_model = seg_model.to(device)
    seg_model.eval()

    # Load mask-only classifier
    clf_model = MROICropClassifier(num_classes=len(GESTURE_LABELS))
    if clf_weights_path is not None and os.path.exists(clf_weights_path) and clf_weights_path.endswith((".pth", ".pt")):
        clf_checkpoint = torch.load(clf_weights_path, map_location=device)
        clf_state = clf_checkpoint.get("model_state_dict", clf_checkpoint)
        try:
            clf_model.load_state_dict(clf_state, strict=True)
            print(f"Loaded mask-only classifier weights from {clf_weights_path}")
        except RuntimeError as e:
            print(f"WARNING: Failed to load classifier weights from {clf_weights_path}: {e}")
            print("Using randomly initialised classifier.")
    else:
        if clf_weights_path and not os.path.exists(clf_weights_path):
            print(f"Classifier weights file not found: {clf_weights_path}")
        elif clf_weights_path and not clf_weights_path.endswith((".pth", ".pt")):
            print(f"PyTorch expects .pth/.pt checkpoints; got {clf_weights_path}")
        print("Using randomly initialised classifier.")

    clf_model = clf_model.to(device)
    clf_model.eval()

    # Pick N random sample indices to visualise
    indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))

    for i, idx in enumerate(indices):
        sample = val_dataset[idx]
        rgbd = sample['input']   # (4, H, W) float
        mask_gt = sample['mask']    # (1, H, W) float
        bbox_gt = sample['bbox']    # (4,) normalised XYXY
        gesture_label = sample['gesture'].item()

        # Add batch dimension and move to device for inference
        x = rgbd.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask = seg_model(x)[0, 0]  # extract (H', W') predicted mask probabilities

        # Convert RGB channels of the input to a uint8 numpy image for PIL
        rgb_np = rgbd[:3].cpu().numpy()  # (3, H, W)
        rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
        rgb_np = np.transpose(rgb_np, (1,2,0)).copy()  # (H, W, 3)
        h, w   = rgb_np.shape[:2]

        # Resize predicted mask to full image resolution for overlay
        # The model outputs at ~half resolution so we bilinear upsample here
        pred_mask_np = pred_mask.cpu().unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        pred_mask_np = F.interpolate(pred_mask_np, size=(h, w), mode="bilinear", align_corners=False)
        pred_mask_np = pred_mask_np[0, 0].numpy()  # back to (H, W)

        # Denormalise GT bounding box from [0,1] to pixel coordinates
        bbox_gt_np = bbox_gt.cpu().numpy() if torch.is_tensor(bbox_gt) else np.asarray(bbox_gt)
        x_min_gt = int(np.clip(bbox_gt_np[0] * w, 0, w - 1))
        y_min_gt = int(np.clip(bbox_gt_np[1] * h, 0, h - 1))
        x_max_gt = int(np.clip(bbox_gt_np[2] * w, 0, w - 1))
        y_max_gt = int(np.clip(bbox_gt_np[3] * h, 0, h - 1))

        # Derive predicted bbox from the post-processed mask (largest connected component)
        pred_mask_binary = (pred_mask_np > 0.5).astype(np.uint8)
        pred_mask_lcc = largest_component_mask(pred_mask_binary)

        # Overlay the POST-PROCESSED mask in red on the RGB image
        vis_img = overlay_mask_on_image(rgb_np, pred_mask_lcc.astype(float), color=(255,0,0), alpha=0.45)
        hand_pixels_pred = np.argwhere(pred_mask_lcc > 0)
        if hand_pixels_pred.size == 0:
            # If model predicts nothing, fall back to GT bbox
            x_min, y_min, x_max, y_max = x_min_gt, y_min_gt, x_max_gt, y_max_gt
            pred_bbox = (x_min, y_min, x_max, y_max)
        else:
            y_coords_p, x_coords_p = hand_pixels_pred[:, 0], hand_pixels_pred[:, 1]
            x_min = int(x_coords_p.min())
            y_min = int(y_coords_p.min())
            x_max = int(x_coords_p.max())
            y_max = int(y_coords_p.max())
            pred_bbox = (x_min, y_min, x_max, y_max)

        # Prepare mask crop and run mask-only classifier
        mask_crop = crop_mask_for_classifier(pred_mask_lcc, target_size=224).to(device)  # (1,1,224,224)
        with torch.no_grad():
            logits = clf_model(mask_crop)
            pred_gesture_idx = int(logits.argmax(dim=1).item())

        # Draw both bboxes on the overlaid image
        vis_img_draw = vis_img.copy()
        draw = ImageDraw.Draw(vis_img_draw)
        # GT bbox in blue -- drawn first so predicted bbox appears on top
        draw_bbox(draw, (x_min_gt, y_min_gt, x_max_gt, y_max_gt), outline=(0, 0, 255), width=3)
        # Predicted bbox in green
        draw_bbox(draw, pred_bbox, outline=(0, 255, 0), width=3)

        # Draw GT and predicted gesture labels as text in the top-left corner
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size=28)
        else:
            font = ImageFont.load_default()  # fallback if DejaVu isn't installed

        gt_name = GESTURE_TO_CLASS.get(gesture_label, str(gesture_label))
        pred_name = GESTURE_TO_CLASS.get(pred_gesture_idx, str(pred_gesture_idx))
        text = f"GT: {gt_name} [{gesture_label}] | Pred: {pred_name} [{pred_gesture_idx}]"
        text_bg  = (0,0,0,160)    # semi-transparent black background for readability
        text_fg  = (0,255,0,255)  # bright green text
        text_pos = (10, 10)
        text_size = draw.textbbox(text_pos, text, font=font)
        draw.rectangle([text_size[0], text_size[1], text_size[2]+8, text_size[3]+4], fill=text_bg)
        draw.text((text_pos[0]+3, text_pos[1]+2), text, fill=text_fg, font=font)

        # Save result image
        save_path = os.path.join(output_dir, f"viz_{i:02}_gt_{gesture_label}_pred_{pred_gesture_idx}.png")
        vis_img_draw.save(save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    # No command-line parsing -- just edit the variables at the top of this file
    if not (DATASET_DIR and isinstance(DATASET_DIR, str) and os.path.isdir(DATASET_DIR)):
        print(f"ERROR: Please set DATASET_DIR correctly at the top of this file. (Current: {DATASET_DIR})")
        exit(1)
    main(
        dataset_dir=DATASET_DIR,
        seg_weights_path=SEG_WEIGHTS_PATH,
        clf_weights_path=CLF_WEIGHTS_PATH,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        seed=RANDOM_SEED,
    )
