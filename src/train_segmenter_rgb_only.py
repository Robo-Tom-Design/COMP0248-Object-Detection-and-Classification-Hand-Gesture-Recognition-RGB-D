"""
Training script for the Hand Analysis segmentation model (RGB-only input).

This is a variant of the standard segmenter training that zeroes out the depth
channel at training time, so the model is forced to learn from RGB alone. The
model architecture is unchanged -- it still takes a 4-channel input -- but
channel 3 (depth) is set to 0 for every batch.

Why do this instead of training a separate 3-channel model? Mainly because it
means we can reuse the same architecture and checkpoint format. At inference
time on a webcam (no real depth sensor), we just zero the depth channel too.

The train/val split is done at the student level -- all frames from a student
are either all in train or all in val, never split across both. This is important
because frames from the same clip are very similar and mixing them would inflate
val accuracy (data leakage). 80% of students go to train, 20% to val.

Usage (from project root), e.g.:

    python src/train_segmenter_rgb_only.py \\
        --data dataset \\
        --epochs 20 \\
        --batch_size 8 \\
        --lr 1e-3 \\
        --run seg_rgb_only \\
        --seed 42

Checkpoints are saved to:
    weights/<run>_best.pt    <- best val loss so far
    weights/<run>_last.pt    <- most recent epoch (for resuming if job gets killed)
"""

import argparse
import os
import sys
from pathlib import Path

import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root to sys.path so we can do `from src.xxx import ...`
# regardless of whether we run with `python src/train_segmenter_rgb_only.py`
# or `python train_segmenter_rgb_only.py` from within src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import GestureDataset  # noqa: E402
from src.model_rgbd_segmenter import RGBDSegmenter as HandAnalysisSegmenterModel, SegmentationLoss  # noqa: E402
from src.utils import collate_fn, compute_iou, compute_dice, AverageMeter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train hand segmentation model (RGB-only input, blank depth)")
    p.add_argument(
        "--data",
        type=str,
        default="dataset",
        help="Path to training dataset root (e.g. dataset or test_dataset)",
    )
    p.add_argument("--epochs",     type=int,   default=20,   help="Number of training epochs")
    p.add_argument("--batch_size", type=int,   default=8,    help="Batch size")
    p.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (Adam)")
    p.add_argument(
        "--run",
        type=str,
        default="segmenter_rgb_only",
        help="Run name for checkpoints (weights/<run>_best.pt, weights/<run>_last.pt)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve the data path
    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / args.data
    if not data_root.exists():
        print(f"ERROR: Data path does not exist: {data_root}")
        sys.exit(1)

    # Set seeds for reproducibility -- important for the train/val split to be consistent
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the full dataset first to get the list of student IDs
    # No transform here -- augmentation is not used for this RGB-only training script
    # (augmentation is handled in train_overnight.py)
    full_dataset = GestureDataset(str(data_root), transform=None)
    if len(full_dataset) == 0:
        print("ERROR: No samples found in dataset.")
        sys.exit(1)

    # Student-level split: shuffle students, use first 80% for train and remaining 20% for val
    # This prevents any student's clips appearing in both sets
    all_students = list(full_dataset.student_ids)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_students)
    num_train_students = max(1, int(0.8 * len(all_students)))
    train_students = sorted(all_students[:num_train_students])
    val_students   = sorted(all_students[num_train_students:])

    # Fallback: if everything ended up in train (single student dataset), move one to val
    if not val_students:
        val_students = [train_students.pop()]

    print(f"Train students: {len(train_students)}, Val students: {len(val_students)}")

    # Create separate dataset objects filtered to each student split
    train_dataset = GestureDataset(str(data_root), transform=None, student_ids=train_students)
    val_dataset   = GestureDataset(str(data_root), transform=None, student_ids=val_students)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,            # shuffle for training -- important for SGD convergence
        num_workers=0,           # 0 workers to avoid multiprocessing issues on the cluster
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),  # faster GPU transfer with pinned memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # no shuffling needed for validation
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Initialise model, loss, and optimiser
    model   = HandAnalysisSegmenterModel().to(device)
    loss_fn = SegmentationLoss(use_dice=True, use_iou=True)  # combined Dice + IoU loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Note: no lr scheduler here, kept simple. The overnight script uses cosine annealing.

    # Set up checkpoint and log paths
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(exist_ok=True)
    best_pt = weights_dir / f"{args.run}_best.pt"
    last_pt = weights_dir / f"{args.run}_last.pt"

    results_dir = PROJECT_ROOT / "results" / args.run
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train_log.csv"

    # Write CSV header for the training log
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "loss", "mean_iou", "mean_dice"])

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # =================== TRAINING PHASE ===================
        model.train()
        train_loss_meter  = AverageMeter()
        train_iou_meter   = AverageMeter()
        train_dice_meter  = AverageMeter()

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [train]",
            leave=False,
        )
        for batch in train_pbar:
            inp  = batch["input"].to(device)  # (B, 4, H, W) -- channels: R,G,B,Depth
            mask = batch["mask"].to(device)   # (B, 1, H, W) -- target segmentation mask

            # RGB-only training: zero out the depth channel (channel index 3)
            # The model still receives a 4-channel tensor, but depth is all zeros
            inp[:, 3:4, ...] = 0.0

            optimizer.zero_grad()
            pred_mask = model(inp)          # (B, 1, H', W') -- predictions (may be half-res)
            loss = loss_fn(pred_mask, mask)  # Dice+IoU loss (handles size mismatch internally)
            loss.backward()
            optimizer.step()

            # Compute IoU and Dice for monitoring -- do this without gradients to save memory
            with torch.no_grad():
                pm_t = pred_mask
                # Resize prediction to ground truth size for metric computation if needed
                if pm_t.shape[2:] != mask.shape[2:]:
                    pm_t = F.interpolate(
                        pm_t,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                # Threshold and convert to numpy for metric computation
                pm = (pm_t.detach().cpu().numpy() > 0.5).astype(np.uint8)
                gm = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
                B = pm.shape[0]
                batch_ious  = [compute_iou(pm[i, 0],  gm[i, 0]) for i in range(B)]
                batch_dices = [compute_dice(pm[i, 0], gm[i, 0]) for i in range(B)]
                mean_iou  = float(np.mean(batch_ious))  if batch_ious  else 0.0
                mean_dice = float(np.mean(batch_dices)) if batch_dices else 0.0

            train_loss_meter.update(loss.item(), inp.size(0))
            train_iou_meter.update(mean_iou, inp.size(0))
            train_dice_meter.update(mean_dice, inp.size(0))

            train_pbar.set_postfix(
                loss=f"{train_loss_meter.avg:.4f}",
                iou=f"{train_iou_meter.avg:.4f}",
                dice=f"{train_dice_meter.avg:.4f}",
            )

        # =================== VALIDATION PHASE ===================
        model.eval()
        val_loss_meter  = AverageMeter()
        val_iou_meter   = AverageMeter()
        val_dice_meter  = AverageMeter()

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{args.epochs} [val]",
            leave=False,
        )
        with torch.no_grad():
            for batch in val_pbar:
                inp  = batch["input"].to(device)
                mask = batch["mask"].to(device)

                # Zero depth channel for val too -- must be consistent with training
                inp[:, 3:4, ...] = 0.0

                pred_mask = model(inp)
                loss = loss_fn(pred_mask, mask)

                pm_t = pred_mask
                if pm_t.shape[2:] != mask.shape[2:]:
                    pm_t = F.interpolate(
                        pm_t,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                pm = (pm_t.cpu().numpy() > 0.5).astype(np.uint8)
                gm = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                B = pm.shape[0]
                batch_ious  = [compute_iou(pm[i, 0],  gm[i, 0]) for i in range(B)]
                batch_dices = [compute_dice(pm[i, 0], gm[i, 0]) for i in range(B)]
                mean_iou  = float(np.mean(batch_ious))  if batch_ious  else 0.0
                mean_dice = float(np.mean(batch_dices)) if batch_dices else 0.0

                val_loss_meter.update(loss.item(), inp.size(0))
                val_iou_meter.update(mean_iou, inp.size(0))
                val_dice_meter.update(mean_dice, inp.size(0))

                val_pbar.set_postfix(
                    loss=f"{val_loss_meter.avg:.4f}",
                    iou=f"{val_iou_meter.avg:.4f}",
                    dice=f"{val_dice_meter.avg:.4f}",
                )

        # Pull final epoch averages out of the meters
        train_loss, train_iou, train_dice = (
            train_loss_meter.avg, train_iou_meter.avg, train_dice_meter.avg,
        )
        val_loss, val_iou, val_dice = (
            val_loss_meter.avg, val_iou_meter.avg, val_dice_meter.avg,
        )

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"TRAIN loss={train_loss:.4f} IoU={train_iou:.4f} Dice={train_dice:.4f} | "
            f"VAL   loss={val_loss:.4f}   IoU={val_iou:.4f}   Dice={val_dice:.4f}"
        )

        # Append this epoch's metrics to the CSV log
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, "train", train_loss, train_iou, train_dice])
            writer.writerow([epoch, "val",   val_loss,   val_iou,   val_dice])

        # Save best checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss":    best_val_loss,
                },
                best_pt,
            )
            print(f"  -> Saved best to {best_pt}")

        # Always save "last" checkpoint so we can resume if training gets interrupted
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss":    best_val_loss,
            },
            last_pt,
        )

    print(f"Training done. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {best_pt}, {last_pt}")


if __name__ == "__main__":
    main()
