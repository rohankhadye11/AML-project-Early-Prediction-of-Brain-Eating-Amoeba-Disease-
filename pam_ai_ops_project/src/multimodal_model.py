"""PyTorch Advanced Multimodal Late Fusion Neural Network Module.

Implements PAMLateFusionNet:
- Branch 1: Pre-trained ResNet-18 adapted for 1-channel grayscale MRI (outputs 64-D embedding).
- Branch 2: 3-layer Tabular MLP for 9 clinical diagnostic anchors (outputs 32-D embedding).
- Fusion Head: Concatenates (64 + 32 = 96-D) representations to output calibrated logits.
"""

from typing import Dict, Tuple
import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger(__name__)


class PAMLateFusionNet(nn.Module):
    """Dual-branch Late Fusion Architecture for Multimodal Clinical Decision Support.

    Architecture Overview:
    ======================
    Image (B, 1, 224, 224) ──> ResNet-18 Backbone ──> Embedding (B, 64) ┐
                                                                         ├── Concatenate (B, 96) ──> Fusion MLP ──> Logit (B, 1)
    Clinical (B, 9)        ──> 3-Layer Tabular MLP──> Embedding (B, 32) ┘
    """

    def __init__(
        self,
        num_clinical_features: int = 9,
        image_embedding_dim: int = 64,
        clinical_embedding_dim: int = 32,
        pretrained: bool = True,
    ) -> None:
        """Initializes the PAMLateFusionNet model.

        Args:
            num_clinical_features: Number of clinical input variables. Defaults to 9.
            image_embedding_dim: Output dimensionality of neuroimaging branch. Defaults to 64.
            clinical_embedding_dim: Output dimensionality of clinical branch. Defaults to 32.
            pretrained: If True, initializes ResNet-18 with ImageNet weights. Defaults to True.
        """
        super(PAMLateFusionNet, self).__init__()

        # =====================================================================
        # BRANCH 1: Neuroimaging Backbone (ResNet-18 Grayscale Adaptation)
        # =====================================================================
        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.image_backbone = resnet18(weights=weights)
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights ({e}). Initializing random ResNet-18.")
            self.image_backbone = resnet18(weights=None)

        # Adapt first convolution layer from 3 RGB channels to 1 Grayscale channel:
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        orig_conv1 = self.image_backbone.conv1
        self.image_backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Preserve pre-trained ImageNet edge/contrast filters by averaging RGB weights
        if pretrained and hasattr(orig_conv1, "weight"):
            with torch.no_grad():
                self.image_backbone.conv1.weight.copy_(
                    orig_conv1.weight.mean(dim=1, keepdim=True)
                )

        # Replace 1000-class ImageNet classification head with 64-D embedding layer
        in_features_resnet = self.image_backbone.fc.in_features  # 512
        self.image_backbone.fc = nn.Linear(
            in_features=in_features_resnet,
            out_features=image_embedding_dim,
            bias=True,
        )
        # Tensor Shape through Branch 1:
        # Input:      (Batch_Size, 1, 224, 224)
        # Conv1-Pool: (Batch_Size, 64, 56, 56)
        # Layer 1-4:  (Batch_Size, 512, 7, 7)
        # AvgPool:    (Batch_Size, 512, 1, 1)
        # Output FC:  (Batch_Size, 64)

        # =====================================================================
        # BRANCH 2: Clinical Tabular Backbone (3-Layer Deep MLP)
        # =====================================================================
        # Architecture: Linear(9 -> 64) -> BatchNorm -> ReLU -> Dropout(0.3)
        #               -> Linear(64 -> 32) -> ReLU -> Linear(32 -> 32)
        self.clinical_backbone = nn.Sequential(
            # Layer 1: Input projection and batch normalization
            nn.Linear(num_clinical_features, 64),               # Shape: (B, 64)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # Layer 2: Latent feature compression
            nn.Linear(64, 32),                                  # Shape: (B, 32)
            nn.ReLU(inplace=True),

            # Layer 3: Final tabular embedding projection
            nn.Linear(32, clinical_embedding_dim),              # Shape: (B, 32)
        )

        # =====================================================================
        # FUSION HEAD: Multimodal Late Integration (96 -> 32 -> 1)
        # =====================================================================
        total_fusion_dim = image_embedding_dim + clinical_embedding_dim  # 64 + 32 = 96
        self.fusion_head = nn.Sequential(
            nn.Linear(total_fusion_dim, 32),                    # Shape: (B, 32)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),                                   # Shape: (B, 1) -> Raw Logits
        )

    def extract_image_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """Extracts 64-dimensional latent embedding from MRI scan.

        Args:
            image: Float32 tensor of shape (B, 1, 224, 224).

        Returns:
            torch.Tensor: Latent embedding of shape (B, 64).
        """
        # image shape: (B, 1, 224, 224) -> output shape: (B, 64)
        return self.image_backbone(image)

    def extract_clinical_embedding(self, clinical: torch.Tensor) -> torch.Tensor:
        """Extracts 32-dimensional latent embedding from tabular features.

        Args:
            clinical: Float32 tensor of shape (B, 9).

        Returns:
            torch.Tensor: Latent embedding of shape (B, 32).
        """
        # clinical shape: (B, 9) -> output shape: (B, 32)
        return self.clinical_backbone(clinical)

    def get_embeddings(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts intermediate latent representations for t-SNE & interpretability.

        Args:
            image: Tensor of shape (B, 1, 224, 224).
            clinical: Tensor of shape (B, 9).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - img_emb: Neuroimaging embedding (B, 64).
                - clin_emb: Clinical tabular embedding (B, 32).
                - fused_emb: Concatenated multimodal representation (B, 96).
        """
        img_emb = self.extract_image_embedding(image)          # (B, 64)
        clin_emb = self.extract_clinical_embedding(clinical)    # (B, 32)
        fused_emb = torch.cat([img_emb, clin_emb], dim=1)      # (B, 96)
        return img_emb, clin_emb, fused_emb

    def forward(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through dual-branch architecture and late fusion head.

        Args:
            image: Batch of 224x224 grayscale MRI slices with shape (B, 1, 224, 224).
            clinical: Batch of 9 clinical diagnostic features with shape (B, 9).

        Returns:
            torch.Tensor: Raw binary classification logits with shape (B, 1).
        """
        # Step 1: Neuroimaging Branch Forward Pass
        # Input shape: (B, 1, 224, 224) -> Output shape: (B, 64)
        img_features = self.image_backbone(image)

        # Step 2: Tabular Clinical Branch Forward Pass
        # Input shape: (B, 9) -> Output shape: (B, 32)
        clin_features = self.clinical_backbone(clinical)

        # Step 3: Late Fusion Concatenation
        # Concatenate along feature dimension -> Shape: (B, 64 + 32) = (B, 96)
        fused_tensor = torch.cat([img_features, clin_features], dim=1)

        # Step 4: Fusion Classification Head
        # Input shape: (B, 96) -> Output shape: (B, 1)
        logits = self.fusion_head(fused_tensor)

        return logits
