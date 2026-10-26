"""PyTorch Multimodal Dataset Module for PAM Clinical Decision Support.

Encapsulates dual-modality clinical patient records:
- 1-channel 224x224 grayscale neuroimaging MRI/CT scans.
- 9-dimensional tabular clinical symptom & biomarker vectors.
- Binary outcomes (0 = Control / Standard Meningitis, 1 = PAM Positive).
Applies stochastic data augmentations for neuroimaging slices.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class PAMMultimodalDataset(Dataset):
    """Dual-branch PyTorch Dataset for neuroimaging MRI and clinical tabular data.

    Pairs raw 224x224 grayscale brain MRI arrays with 9-dimensional RFE-optimized
    clinical features and binary diagnostic labels.

    Attributes:
        clinical_data: Float32 tensor of shape (N, 9).
        images: Float32 tensor or raw array of shape (N, 224, 224).
        labels: Float32 tensor of shape (N, 1).
        transform: Torchvision transform pipeline applied to MRI slices.
    """

    def __init__(
        self,
        clinical_data: Union[pd.DataFrame, np.ndarray],
        images: Union[np.ndarray, List[np.ndarray], List[str]],
        labels: Union[pd.Series, np.ndarray, List[int]],
        is_training: bool = True,
        transform: Optional[T.Compose] = None,
    ) -> None:
        """Initializes the PAMMultimodalDataset.

        Args:
            clinical_data: 2D array or DataFrame with exactly 9 clinical features.
            images: 3D array of shape (N, 224, 224), list of 2D arrays, or list of image paths.
            labels: 1D array or Series of binary outcomes (0 or 1).
            is_training: If True and transform is None, applies training augmentations
                (random horizontal flips and rotation ±10°). Defaults to True.
            transform: Optional custom torchvision transform pipeline.

        Raises:
            ValueError: If sample counts across modalities do not match or features != 9.
        """
        # 1. Parse and validate Clinical Features (N, 9)
        if isinstance(clinical_data, pd.DataFrame):
            self.clinical_features_ = list(clinical_data.columns)
            clin_arr = clinical_data.values.astype(np.float32)
        else:
            clin_arr = np.asarray(clinical_data, dtype=np.float32)
            self.clinical_features_ = [f"clinical_feat_{i}" for i in range(clin_arr.shape[1])]

        if clin_arr.ndim != 2 or clin_arr.shape[1] != 9:
            raise ValueError(f"Expected clinical features with shape (N, 9), received {clin_arr.shape}")

        # 2. Parse and validate Target Labels (N, 1)
        if isinstance(labels, (pd.Series, pd.DataFrame)):
            lbl_arr = labels.values.astype(np.float32).reshape(-1, 1)
        else:
            lbl_arr = np.asarray(labels, dtype=np.float32).reshape(-1, 1)

        # 3. Parse and validate Image Scans (N, 224, 224)
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            self.image_paths = images
            self.images = None
            n_samples = len(images)
        else:
            self.image_paths = None
            img_arr = np.asarray(images)
            if img_arr.ndim == 3:  # (N, H, W)
                if img_arr.shape[1:] != (224, 224):
                    raise ValueError(f"Expected image resolution (224, 224), received {img_arr.shape[1:]}")
            elif img_arr.ndim == 4 and img_arr.shape[1] == 1:  # (N, 1, H, W)
                img_arr = img_arr.squeeze(1)
            else:
                raise ValueError(f"Invalid image array shape {img_arr.shape}")
            self.images = img_arr
            n_samples = len(img_arr)

        if len(clin_arr) != n_samples or len(lbl_arr) != n_samples:
            raise ValueError(
                f"Sample count mismatch: clinical={len(clin_arr)}, images={n_samples}, labels={len(lbl_arr)}"
            )

        self.clinical_tensors = torch.tensor(clin_arr, dtype=torch.float32)
        self.label_tensors = torch.tensor(lbl_arr, dtype=torch.float32)
        self.is_training = is_training

        # 4. Set up Image Transforms & Augmentations
        if transform is not None:
            self.transform = transform
        else:
            if self.is_training:
                # Augmentation pipeline for training set
                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=10),
                    T.ToTensor(),  # Maps [0, 255] uint8 -> [0.0, 1.0] float32, adds channel dim (1, 224, 224)
                    T.Normalize(mean=[0.5], std=[0.5]),  # Standardizes to [-1.0, 1.0]
                ])
            else:
                # Deterministic preprocessing for validation / test set
                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5]),
                ])

    def __len__(self) -> int:
        """Returns total number of multimodal records."""
        return len(self.clinical_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves multimodal triplet for index `idx`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - image_tensor: Float32 tensor of shape (1, 224, 224).
                - clinical_tensor: Float32 tensor of shape (9,).
                - label_tensor: Float32 tensor of shape (1,).
        """
        # Load raw 2D grayscale slice
        if self.images is not None:
            raw_img = self.images[idx]
        else:
            # Read from disk if paths were supplied
            raw_img = Image.open(self.image_paths[idx]).convert("L")
            raw_img = np.array(raw_img)

        # Ensure correct uint8 format for PIL transform
        if raw_img.dtype != np.uint8:
            if raw_img.max() <= 1.0:
                raw_img = (raw_img * 255.0).astype(np.uint8)
            else:
                raw_img = np.clip(raw_img, 0, 255).astype(np.uint8)

        # Apply augmentation and channel formatting -> (1, 224, 224)
        image_tensor = self.transform(raw_img)

        clinical_tensor = self.clinical_tensors[idx]  # Shape: (9,)
        label_tensor = self.label_tensors[idx]        # Shape: (1,)

        return image_tensor, clinical_tensor, label_tensor
