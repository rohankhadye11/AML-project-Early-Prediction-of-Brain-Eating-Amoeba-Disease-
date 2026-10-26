"""Neuroimaging Explainability Module for PAM Clinical Decision Support System.

Implements MRIExplainer using Gradient-Weighted Class Activation Mapping (Grad-CAM)
adapted for the multimodal dual-input PAMLateFusionNet architecture.

Identifies discriminative spatial regions in 224x224 brain MRI scans (e.g., hyperintense
basal meningitis edema foci) that drive the model's diagnostic risk prediction.
"""

from typing import List, Optional, Tuple, Union
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None
    show_cam_on_image = None

logger = logging.getLogger(__name__)


class DualInputVisionWrapper(nn.Module):
    """Adapter wrapping a multimodal model to expose a single-input vision interface for Grad-CAM.

    Grad-CAM expects a callable mapping `image_tensor -> logits`. In our dual-branch
    architecture, forward requires `(image, clinical)`. This wrapper binds a patient's
    clinical tabular vector as a constant context, allowing gradients to propagate exclusively
    through the neuroimaging backbone (ResNet-18).
    """

    def __init__(self, full_model: nn.Module, clinical_tensor: torch.Tensor) -> None:
        """Initializes the vision wrapper adapter.

        Args:
            full_model: Trained multimodal PyTorch model (e.g., PAMLateFusionNet).
            clinical_tensor: Patient's clinical features of shape (1, 9) or (B, 9).
        """
        super().__init__()
        self.full_model = full_model
        # Store clinical tensor as buffer to preserve device location
        self.register_buffer("fixed_clinical", clinical_tensor.detach().clone())

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Executes forward pass with fixed clinical context.

        Args:
            image_tensor: Neuroimaging slice tensor of shape (B, 1, 224, 224).

        Returns:
            torch.Tensor: Binary classification logit of shape (B, 1).
        """
        batch_size = image_tensor.shape[0]
        if self.fixed_clinical.shape[0] != batch_size:
            clin_batch = self.fixed_clinical.expand(batch_size, -1)
        else:
            clin_batch = self.fixed_clinical
        return self.full_model(image_tensor, clin_batch)


class MRIExplainer:
    """Gradient-Weighted Class Activation Mapping (Grad-CAM) explainer for brain MRI scans.

    Extracts coarse 2D activation heatmaps from the final convolutional layer of ResNet-18
    (layer4[-1]) to visually corroborate neuropathological amoebic lesions.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_cuda: bool = False,
    ) -> None:
        """Initializes the MRI explainer.

        Args:
            model: Trained PAMLateFusionNet instance.
            target_layer: Specific Conv2d / Bottleneck layer to inspect. If None,
                defaults to `layer4[-1]` of the vision branch.
            use_cuda: Whether to execute CAM computations on GPU. Defaults to False.
        """
        if GradCAM is None or show_cam_on_image is None:
            raise ImportError(
                "pytorch_grad_cam is required. Please install it via: pip install grad-cam"
            )

        self.model = model
        self.model.eval()
        self.use_cuda = use_cuda and torch.cuda.is_available()

        # Resolve target convolutional layer in ResNet-18 backbone
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "vision_branch") and hasattr(model.vision_branch, "layer4"):
            self.target_layer = model.vision_branch.layer4[-1]
        elif hasattr(model, "image_backbone") and hasattr(model.image_backbone, "layer4"):
            self.target_layer = model.image_backbone.layer4[-1]
        else:
            raise AttributeError(
                "Could not automatically resolve ResNet-18 layer4[-1]. "
                "Ensure model has 'vision_branch' or 'image_backbone' attribute."
            )

        logger.info(f"Initialized MRIExplainer with target layer: {self.target_layer}")

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        clinical_tensor: torch.Tensor,
        original_image_array: np.ndarray,
        output_path: str = "gradcam_output.png",
    ) -> np.ndarray:
        """Computes Grad-CAM activations and saves the anatomical heatmap overlay.

        Args:
            image_tensor: Normalized image tensor of shape (1, 1, 224, 224).
            clinical_tensor: Clinical features tensor of shape (1, 9).
            original_image_array: Original image array for visualization, shape (224, 224)
                or (224, 224, 3) with intensities in [0, 255] or [0.0, 1.0].
            output_path: Destination file path for the overlaid image. Defaults to 'gradcam_output.png'.

        Returns:
            np.ndarray: Blended RGB visualization array of shape (224, 224, 3) in uint8 [0, 255].
        """
        # Step 1: Wrap multimodal model to bind clinical context
        wrapper = DualInputVisionWrapper(self.model, clinical_tensor)
        wrapper.eval()

        # Step 2: Determine target layer within the wrapped instance
        if hasattr(wrapper.full_model, "vision_branch"):
            target_layers = [wrapper.full_model.vision_branch.layer4[-1]]
        else:
            target_layers = [wrapper.full_model.image_backbone.layer4[-1]]

        # Step 3: Instantiate GradCAM
        with GradCAM(
            model=wrapper,
            target_layers=target_layers,
        ) as cam:
            # Generate raw grayscale activation map (shape: [1, 224, 224])
            grayscale_cam = cam(input_tensor=image_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]  # Shape: (224, 224)

        # Step 4: Prepare background anatomical image (Normalized RGB float32 in [0, 1])
        if original_image_array.ndim == 2:
            rgb_base = cv2.cvtColor(original_image_array, cv2.COLOR_GRAY2RGB)
        elif original_image_array.shape[2] == 1:
            rgb_base = cv2.cvtColor(original_image_array, cv2.COLOR_GRAY2RGB)
        else:
            rgb_base = original_image_array.copy()

        if rgb_base.dtype == np.uint8 or rgb_base.max() > 1.0:
            rgb_base = rgb_base.astype(np.float32) / 255.0
        else:
            rgb_base = rgb_base.astype(np.float32)

        # Ensure spatial dimension alignment
        if rgb_base.shape[:2] != (224, 224):
            rgb_base = cv2.resize(rgb_base, (224, 224))

        # Step 5: Overlay heatmap using jet colormap
        visualization = show_cam_on_image(rgb_base, grayscale_cam, use_rgb=True)

        # Step 6: Save visualization to disk
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Convert RGB to BGR for cv2.imwrite
        bgr_vis = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_file), bgr_vis)
        logger.info(f"Grad-CAM neuroimaging overlay successfully saved to: {out_file.resolve()}")

        return visualization
