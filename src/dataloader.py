import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import Image, Mask, BoundingBoxes

# Gesture class mapping — these match the folder names in the dataset (G01_call etc.)
# The labels are 0-indexed integers which is what the classifier expects
# I chose this specific ordering to match the coursework spec, not sure if order matters
# but keeping it consistent anyway
GESTURE_LABELS = {
    'call': 0,
    'dislike': 1,
    'like': 2,
    'ok': 3,
    'one': 4,
    'palm': 5,
    'peace': 6,
    'rock': 7,
    'stop': 8,
    'three': 9
}

# Reverse mapping so we can go from integer back to human-readable name
# useful for printing out results and visualisation
GESTURE_TO_CLASS = {v: k for k, v in GESTURE_LABELS.items()}


class GestureDataset(Dataset):
    def __init__(self, root_dir, transform=None, student_ids=None):
        """
        Main dataset class for loading hand gesture data.

        The dataset folder structure is expected to be:
            root_dir/
              STUDENT_ID/
                G0X_gesture_name/
                  CLIP_ID/
                    rgb/          <- colour frames (.png)
                    depth/        <- colourised depth images (not actually used much)
                    depth_raw/    <- raw depth as .npy float arrays (what we actually use)
                    annotation/   <- binary hand masks (.png)

        We glob for annotation files first and then reconstruct the other paths from that.
        This means if an annotation exists but the rgb or depth_raw is missing, we skip
        that sample -- better to skip than crash halfway through training.

        The transform should be a torchvision v2 transform that knows how to handle
        tv_tensors (Image, Mask, BoundingBoxes) so that geometric transforms get
        applied consistently to everything. If transform=None we just return the
        raw tensors without augmentation, which is what we want for validation/test.

        student_ids: optional list of student folder names to filter to -- this is
        how we do the train/val split without mixing students across sets.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform  # Must accept and return tv_tensors (Image, Mask, BoundingBoxes)
        self.samples = []
        self.invalid_gestures = set()  # track any folder names we couldn't parse, for debugging
        self._student_filter = set(student_ids) if student_ids is not None else None

        # Glob all annotation images at once -- much faster than recursively walking dirs
        mask_files = list(self.root_dir.glob("*/*/*/annotation/*"))
        allowed_exts = {'.png', '.jpg', '.jpeg'}

        for mask_path in mask_files:
            filename = mask_path.name
            # skip hidden files (macOS .DS_Store etc.) and non-image files
            if filename.startswith('.') or mask_path.suffix.lower() not in allowed_exts:
                continue

            # Walk back up the tree to get the gesture directory name
            clip_dir = mask_path.parent.parent
            gesture_dir = clip_dir.parent
            gesture_dir_name = gesture_dir.name

            try:
                # Folder names are like "G01_call" or "G02_dislike" -- we split on first underscore
                # and take the part after it. Some folders might just be the gesture name directly.
                if '_' in gesture_dir_name:
                    gesture_name = gesture_dir_name.split('_', 1)[1].lower()
                else:
                    gesture_name = gesture_dir_name.lower()

                # If the gesture name isn't in our mapping, skip and record it
                if gesture_name not in GESTURE_LABELS:
                    self.invalid_gestures.add(gesture_name)
                    continue
                gesture_label = GESTURE_LABELS[gesture_name]
            except Exception as e:
                print(f"Warning: Could not parse gesture from {gesture_dir_name}: {e}")
                continue

            # Build expected paths for the other modalities
            rgb_path = clip_dir / "rgb" / filename
            depth_path = clip_dir / "depth" / filename
            depth_raw_path = clip_dir / "depth_raw" / f"{mask_path.stem}.npy"

            # Only include sample if all three required files exist
            # We need rgb for colour input, depth_raw for the depth channel,
            # and the mask is already the annotation we're iterating over
            if rgb_path.exists() and depth_path.exists() and depth_raw_path.exists():
                # The student ID is just the top-level folder name under root
                student_id = mask_path.relative_to(self.root_dir).parts[0]

                # If we have a student filter active, skip students not in it
                if self._student_filter is not None and student_id not in self._student_filter:
                    continue

                self.samples.append({
                    "rgb": rgb_path,
                    "depth": depth_path,
                    "depth_raw": depth_raw_path,
                    "mask": mask_path,
                    "gesture": gesture_label,
                    "gesture_name": gesture_name,
                    "student_id": student_id,
                })

        # Build sorted list of unique student IDs actually present in our sample list
        self.student_ids = sorted(set(s["student_id"] for s in self.samples))
        print(f"Dataset loaded! Found {len(self.samples)} valid annotated frames across {len(self.student_ids)} students.")
        if self.invalid_gestures:
            print(f"Warning: Found {len(self.invalid_gestures)} invalid gesture labels: {self.invalid_gestures}")

    def __len__(self):
        """
        Returns total number of samples. Called by DataLoader to know how many batches to make.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads and returns one sample from disk. This is called by the DataLoader
        for every individual item in a batch.

        Returns a dict with:
            'input':   (4, H, W) float tensor -- RGB concatenated with normalised depth
            'mask':    (1, H, W) float tensor -- binary hand mask, values in {0.0, 1.0}
            'bbox':    (4,) float tensor      -- normalised XYXY bounding box derived from mask
            'gesture': scalar long tensor     -- gesture class index (0-9)
        """
        paths = self.samples[idx]

        # Load the RGB image -- convert to RGB explicitly in case any are RGBA or grayscale
        rgb_img = PILImage.open(paths["rgb"]).convert("RGB")
        mask_img = PILImage.open(paths["mask"])

        # Convert mask to numpy so we can find the hand pixels for bbox computation
        mask_np = np.array(mask_img)

        # Compute bounding box from the mask pixels
        # We treat any pixel value > 0 as belonging to the hand (some masks use 255, some use 1)
        hand_pixels = np.argwhere(mask_np > 0)

        if hand_pixels.size == 0:
            # Edge case: empty mask, use full image as fallback bounding box
            # This shouldn't happen often but better than crashing
            h, w = mask_np.shape[-2:]
            x_min, y_min, x_max, y_max = 0, 0, 1, 1
            bbox = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        else:
            # argwhere returns (row, col) so y is first, x is second
            y_coords, x_coords = hand_pixels[:, 0], hand_pixels[:, 1]
            x_min, y_min = x_coords.min(), y_coords.min()
            x_max, y_max = x_coords.max(), y_coords.max()
            # TV-Tensors use (x_min, y_min, x_max, y_max) format -- XYXY
            bbox = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        # Always shape [1, 4] -- the extra dimension is because BoundingBoxes can hold multiple boxes

        # Normalise bbox coordinates to [0, 1] relative to image width/height
        # We do this before the transform so after the transform we recompute from the canvas size
        h, w = mask_np.shape
        bbox_norm = bbox.clone()
        bbox_norm[:, [0, 2]] = bbox_norm[:, [0, 2]] / float(w)  # x_min, x_max divided by width
        bbox_norm[:, [1, 3]] = bbox_norm[:, [1, 3]] / float(h)  # y_min, y_max divided by height

        # Load depth: raw .npy float array, clip to [0, 1500mm] then normalise to [0, 1]
        # 1500mm was chosen as a reasonable max arm reach distance
        depth_raw = np.load(paths["depth_raw"]).astype(np.float32)
        depth_np = np.clip(depth_raw, 0, 1500) / 1500.0
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)  # add channel dim -> (1, H, W)

        gesture_label = torch.tensor(paths["gesture"], dtype=torch.long)

        # Wrap tensors as tv_tensors so that v2 transforms apply correctly to each type
        # Image gets colour transforms, Mask gets geometric transforms (no interpolation artefacts),
        # BoundingBoxes get coordinate transforms, and depth we wrap as Mask to get the same
        # geometric transforms applied to it
        rgb_tensor = Image(v2.functional.pil_to_tensor(rgb_img))  # (3, H, W) uint8
        mask_tensor = Mask(torch.from_numpy(mask_np).unsqueeze(0).float())  # (1, H, W)
        bboxes_tensor = BoundingBoxes(bbox, format="XYXY", canvas_size=(h, w))
        depth_tv = Mask(depth_tensor)  # wrap depth as Mask so it gets the same spatial transforms

        # Apply transform if one was provided (e.g. random flips, crops for training)
        # We pass everything as a single dict so the transforms can handle them together
        if self.transform is not None:
            tvtensors = {
                'image': rgb_tensor,
                'mask': mask_tensor,
                'bboxes': bboxes_tensor,
                'depth': depth_tv
            }
            tvtensors = self.transform(tvtensors)
            rgb_tensor = tvtensors['image']
            mask_tensor = tvtensors['mask']
            bboxes_tensor = tvtensors['bboxes']
            depth_tensor = tvtensors['depth']

        # Make sure depth is a plain float tensor after potentially being unwrapped from Mask
        if isinstance(depth_tensor, Mask):
            depth_tensor = depth_tensor.data.float()
        elif isinstance(depth_tensor, torch.Tensor):
            depth_tensor = depth_tensor.float()

        # Recompute the normalised bbox using the canvas_size from the (possibly transformed)
        # bboxes_tensor -- this is important because transforms like crop change canvas size
        bbox_xyxy = bboxes_tensor.clone()
        bbox_xyxy_norm = bbox_xyxy.clone().float()
        canvas_h, canvas_w = bboxes_tensor.canvas_size
        bbox_xyxy_norm[:, [0, 2]] = bbox_xyxy[:, [0, 2]] / float(canvas_w)
        bbox_xyxy_norm[:, [1, 3]] = bbox_xyxy[:, [1, 3]] / float(canvas_h)
        # Squeeze to (4,) and clamp to [0,1] -- sometimes augmentation can push coords slightly outside
        bbox_xyxy_norm = bbox_xyxy_norm.squeeze(0).clamp(0.0, 1.0)

        # Extract raw mask data from Mask wrapper
        if isinstance(mask_tensor, Mask):
            mask_tensor = mask_tensor.data.float()
        elif isinstance(mask_tensor, torch.Tensor):
            mask_tensor = mask_tensor.float()

        # Normalise RGB to float [0, 1] -- it comes out of pil_to_tensor as uint8
        if rgb_tensor.dtype == torch.uint8:
            rgb_tensor = rgb_tensor.float() / 255.

        # In rare cases the spatial dimensions of rgb and depth might differ (e.g. if a transform
        # was applied to only one of them) -- resize depth to match rgb if needed
        if rgb_tensor.shape[1:] != depth_tensor.shape[1:]:
            import torch.nn.functional as F
            depth_tensor = F.interpolate(depth_tensor.unsqueeze(0), size=rgb_tensor.shape[1:], mode="bilinear", align_corners=False).squeeze(0)

        # Concatenate RGB (3 channels) + Depth (1 channel) -> (4, H, W) early-fusion input
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)

        # Return everything the training loop needs
        return {
            'input': rgbd_tensor,   # (4, H, W) float -- the model input
            'mask': mask_tensor,    # (1, H, W) float -- segmentation target
            'bbox': bbox_xyxy_norm, # (4,) float -- detection target (normalised XYXY)
            'gesture': gesture_label  # () long -- classification target
        }
