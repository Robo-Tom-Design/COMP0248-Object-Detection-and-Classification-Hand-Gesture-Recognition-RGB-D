"""
Inference and evaluation script for the Hand Analysis model.

This is a standalone inference wrapper class that can be used to run the
segmenter on individual images or batches of images. It was written early in
the project when the model was a multi-task network (segmentation + detection
+ classification), so it still has some references to those outputs -- most
of them won't work with the current segmentation-only architecture.

Kept here for reference but the more up-to-date evaluation is in evaluate.py.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Note: these imports use relative names (no 'src.' prefix) which means this
# script needs to be run from inside the src/ directory to work properly.
# That's a bit annoying -- the other scripts all support running from project root.
from model_segmenter import HandAnalysisSegmenterModel
from utils import process_mask, process_bbox, compute_iou, compute_dice


class HandAnalysisInference:
    """
    Wraps the trained hand analysis model for single-image and batch inference.

    Loads a checkpoint, runs inference, post-processes outputs, and can produce
    visualisations. Originally designed for the multi-task version of the model
    (segmentation + detection + classification) but the structure is reusable.
    """

    # Class-level label lookup -- same as in dataloader.py
    GESTURE_LABELS = [
        'call', 'dislike', 'like', 'ok', 'one',
        'palm', 'peace', 'rock', 'stop', 'three'
    ]

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Load the model from a checkpoint file and prepare it for inference.

        Args:
            checkpoint_path: Path to a saved .pth or .pt checkpoint file
            device:          'cuda' or 'cpu' -- will fall back to CPU if CUDA unavailable
        """
        self.device = device

        print(f"Loading segmentation model from {checkpoint_path}...")
        self.model = HandAnalysisSegmenterModel()

        # Load saved weights into the model -- map_location handles GPU->CPU transfer
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()  # eval mode: BatchNorm uses running mean/var, no Dropout

        print("Model loaded successfully")

    def predict(self, image_path, return_raw=False):
        """
        Run inference on a single RGB image file.

        Loads the image, resizes to the model's expected input size (640x480),
        runs the forward pass, and post-processes the outputs.

        Note: This only works with the old multi-task model that returned
        a dict with 'mask', 'bbox', and 'gesture'. The current segmenter
        only returns a mask tensor -- so bbox and gesture outputs won't be
        available with the updated model.

        Args:
            image_path: Path string or Path object to an image file
            return_raw: If True return raw model outputs before post-processing.
                        If False (default) return post-processed results.

        Returns:
            Dict with keys:
                'mask':               post-processed binary mask (H, W) or raw prob map
                'bbox':               pixel bbox (x1, y1, x2, y2)
                'gesture':            gesture name string
                'gesture_class':      integer class index
                'gesture_confidence': softmax confidence float
                'mask_raw':           raw predicted mask probabilities (H, W)
        """
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size  # (W, H) -- saved but not used currently

        # Resize to the model's expected input size
        image = image.resize((640, 480))

        # Convert to (1, 3, H, W) float tensor normalised to [0, 1]
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Run inference without computing gradients (saves memory and is faster)
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Extract each output -- this expects the old multi-task dict format
        mask             = outputs['mask'][0, 0].cpu().numpy()
        bbox             = outputs['bbox'][0].cpu().numpy()
        gesture_logits   = outputs['gesture'][0].cpu().numpy()
        gesture_class    = np.argmax(gesture_logits)
        # Softmax confidence for the predicted class
        gesture_confidence = np.exp(gesture_logits[gesture_class]) / np.exp(gesture_logits).sum()

        if return_raw:
            return {
                'mask':               mask,
                'bbox':               bbox,
                'gesture':            gesture_class,
                'gesture_confidence': gesture_confidence,
                'gesture_logits':     gesture_logits
            }

        # Post-process mask and bbox using helpers from utils.py
        mask_processed = process_mask(mask, threshold=0.5)  # removes small noise components
        bbox_processed = process_bbox(bbox, (480, 640))     # convert normalised to pixel coords

        return {
            'mask':               mask_processed,
            'bbox':               bbox_processed,
            'gesture':            self.GESTURE_LABELS[gesture_class],
            'gesture_class':      gesture_class,
            'gesture_confidence': float(gesture_confidence),
            'mask_raw':           mask  # include raw for visualisation
        }

    def visualize_predictions(self, image_path, predictions, save_path=None):
        """
        Create a 3-panel visualisation of inference results on a single image.

        Shows:
            Panel 1: Original image
            Panel 2: Raw segmentation mask probability map
            Panel 3: Original image with predicted bounding box overlaid, and
                     the predicted gesture name + confidence in the title.

        Args:
            image_path:  Path to the original image file
            predictions: Output dict from predict()
            save_path:   Optional path to save the figure (e.g. 'vis/pred.png')
                         If None, plt.show() is called (won't work over SSH).
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize((640, 480))
        image_array = np.array(image)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original RGB frame
        axes[0].imshow(image_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Panel 2: Raw mask probability map -- shows where the model thinks the hand is
        axes[1].imshow(predictions['mask_raw'], cmap='gray')
        axes[1].set_title(f"Segmentation Mask")
        axes[1].axis('off')

        # Panel 3: Bounding box + gesture label on original image
        axes[2].imshow(image_array)
        x1, y1, x2, y2 = predictions['bbox']
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        axes[2].add_patch(rect)
        gesture = predictions['gesture']
        conf    = predictions['gesture_confidence']
        axes[2].set_title(f"Detection: {gesture}\nConfidence: {conf:.2%}")
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def batch_predict(self, image_dir, output_dir=None):
        """
        Run inference on all PNG/JPG images in a directory.

        Processes them in sorted order and prints results as it goes.
        Any images that fail (corrupt files, wrong size etc.) are skipped
        with an error message printed.

        Args:
            image_dir:  Directory containing image files
            output_dir: Optional directory to save per-image visualisations

        Returns:
            List of prediction dicts, one per successfully processed image.
        """
        image_dir        = Path(image_dir)
        predictions_list = []

        # Collect all images in directory
        image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

        print(f"Processing {len(image_files)} images...")
        for image_path in image_files:
            try:
                predictions = self.predict(str(image_path))
                predictions['image_path'] = str(image_path)
                predictions_list.append(predictions)
                print(f"  {image_path.name}: {predictions['gesture']} ({predictions['gesture_confidence']:.2%})")
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")

        return predictions_list


def main():
    """
    Example usage of the inference class.

    In practice we mostly use evaluate.py for batch evaluation -- this script
    is more of a demo/development helper.
    """
    checkpoint_path = Path('checkpoints/best_model.pth')

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference = HandAnalysisInference(str(checkpoint_path), device=device)

    # Example usage (uncomment and fill in a real path to test):
    # image_path = 'path/to/image.jpg'
    # predictions = inference.predict(image_path)
    # print(f"Gesture: {predictions['gesture']} ({predictions['gesture_confidence']:.2%})")
    # inference.visualize_predictions(image_path, predictions, save_path='prediction.png')

    print("\nInference module ready. Use the following to make predictions:")
    print("\n  from inference import HandAnalysisInference")
    print("  inference = HandAnalysisInference('checkpoints/best_model.pth')")
    print("  predictions = inference.predict('image.jpg')")


if __name__ == '__main__':
    main()
