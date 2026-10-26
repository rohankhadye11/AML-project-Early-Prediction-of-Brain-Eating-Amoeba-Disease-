"""Clinical Tabular Explainability Module for PAM Clinical Decision Support System.

Implements ClinicalSHAPExplainer utilizing SHAP (SHapley Additive exPlanations)
adapted for the multimodal dual-input PAMLateFusionNet architecture.

Calculates patient-specific Shapley attribution values across the 9 clinical diagnostic
features conditioned on a fixed neuroimaging slice context.
"""

from typing import List, Optional, Tuple, Union
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    import shap
except ImportError:
    shap = None

logger = logging.getLogger(__name__)


class DualInputClinicalWrapper(nn.Module):
    """Adapter wrapping a multimodal model to expose a single-input tabular interface for SHAP.

    SHAP explainers (DeepExplainer / GradientExplainer) expect a callable mapping
    `clinical_tensor -> output_scalar`. This wrapper locks the patient's 224x224 MRI slice
    as a constant latent context, enabling Shapley value attribution specifically across
    the 9 bedside clinical parameters.
    """

    def __init__(self, full_model: nn.Module, fixed_image_tensor: torch.Tensor) -> None:
        """Initializes the clinical wrapper adapter.

        Args:
            full_model: Trained multimodal PyTorch model (e.g., PAMLateFusionNet).
            fixed_image_tensor: Fixed MRI slice tensor of shape (1, 1, 224, 224).
        """
        super().__init__()
        self.full_model = full_model

        # Ensure non-inplace ReLU activations to support PyTorch autograd backward hooks in SHAP
        for module in self.full_model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Extract fixed neuroimaging embedding (B=1, 64) for the patient context
        with torch.no_grad():
            if hasattr(full_model, "extract_image_embedding"):
                img_emb = full_model.extract_image_embedding(fixed_image_tensor)
            elif hasattr(full_model, "vision_branch"):
                img_emb = full_model.vision_branch(fixed_image_tensor)
            elif hasattr(full_model, "image_backbone"):
                img_emb = full_model.image_backbone(fixed_image_tensor)
            else:
                raise AttributeError("Could not resolve image embedding extractor on model.")

        self.register_buffer("fixed_img_emb", img_emb.detach().clone())

    def forward(self, clinical_tensor: torch.Tensor) -> torch.Tensor:
        """Executes forward pass with fixed neuroimaging context.

        Args:
            clinical_tensor: Clinical features tensor of shape (B, 9).

        Returns:
            torch.Tensor: Calibrated PAM probability [0.0, 1.0] of shape (B, 1).
        """
        batch_size = clinical_tensor.shape[0]
        # Propagate through tabular branch
        clin_features = self.full_model.clinical_backbone(clinical_tensor)

        # Concatenate with locked image embedding across batch
        fused = torch.cat([self.fixed_img_emb.expand(batch_size, -1), clin_features], dim=1)

        # Output calibrated probability through late fusion head
        logits = self.full_model.fusion_head(fused)
        return torch.sigmoid(logits)


class ClinicalSHAPExplainer:
    """SHAP (SHapley Additive exPlanations) engine for multimodal clinical tabular inputs.

    Measures the marginal contribution of each symptom, lab parameter, and demographic
    variable toward the predicted risk of Primary Amoebic Meningoencephalitis (PAM).
    """

    def __init__(
        self,
        model: nn.Module,
        background_clinical: torch.Tensor,
        background_image: Optional[torch.Tensor] = None,
    ) -> None:
        """Initializes the clinical SHAP explainer with a reference background cohort.

        Args:
            model: Trained multimodal PyTorch model.
            background_clinical: Reference dataset of baseline clinical vectors, shape (100, 9).
            background_image: Reference MRI slice for background conditioning, shape (1, 1, 224, 224).
                If None, defaults to zero-valued baseline slice.
        """
        if shap is None:
            raise ImportError("shap is required. Please install it via: pip install shap")

        self.model = model
        self.model.eval()

        # Ensure background clinical is a float tensor of shape (N, 9)
        if isinstance(background_clinical, np.ndarray):
            self.background_clinical = torch.from_numpy(background_clinical).float()
        else:
            self.background_clinical = background_clinical.float()

        if background_image is not None:
            self.default_image = background_image.float()
        else:
            self.default_image = torch.zeros((1, 1, 224, 224), dtype=torch.float32)

        logger.info(
            f"Initialized ClinicalSHAPExplainer with background cohort of {len(self.background_clinical)} records."
        )

    def explain_patient(
        self,
        patient_clinical_tensor: torch.Tensor,
        feature_names: List[str],
        patient_image_tensor: Optional[torch.Tensor] = None,
        output_path: str = "shap_output.png",
    ) -> np.ndarray:
        """Calculates Shapley values for a patient and saves a publication-grade attribution chart.

        Args:
            patient_clinical_tensor: Query patient clinical vector of shape (1, 9) or (9,).
            feature_names: List of 9 feature names corresponding to the vector columns.
            patient_image_tensor: Query patient MRI scan of shape (1, 1, 224, 224). If None,
                uses the default baseline image.
            output_path: Destination path for the generated chart. Defaults to 'shap_output.png'.

        Returns:
            np.ndarray: 1D array of 9 SHAP attribution values for the query patient.
        """
        # Step 1: Format query tensors
        if isinstance(patient_clinical_tensor, np.ndarray):
            query_clin = torch.from_numpy(patient_clinical_tensor).float()
        else:
            query_clin = patient_clinical_tensor.float()

        if query_clin.ndim == 1:
            query_clin = query_clin.unsqueeze(0)  # Shape: (1, 9)

        target_img = patient_image_tensor if patient_image_tensor is not None else self.default_image

        # Step 2: Wrap model to freeze the patient's specific MRI scan
        wrapped_model = DualInputClinicalWrapper(self.model, target_img)
        wrapped_model.eval()

        # Step 3: Compute SHAP values via DeepExplainer or GradientExplainer fallback
        try:
            logger.info("Computing attributions using shap.DeepExplainer...")
            explainer = shap.DeepExplainer(wrapped_model, self.background_clinical)
            raw_shap = explainer.shap_values(query_clin)
        except Exception as e:
            logger.warning(
                f"shap.DeepExplainer hook negotiation encountered ({e}). "
                "Switching to shap.GradientExplainer for robust autograd attribution..."
            )
            explainer = shap.GradientExplainer(wrapped_model, self.background_clinical)
            raw_shap = explainer.shap_values(query_clin)

        # Step 4: Extract 1D array of attributions (handling various SHAP versions)
        if isinstance(raw_shap, list):
            shap_vals = np.array(raw_shap[0]).flatten()
        elif isinstance(raw_shap, np.ndarray):
            shap_vals = raw_shap.flatten()
        else:
            shap_vals = np.array(raw_shap.values).flatten()

        # Match dimensions exactly
        if len(shap_vals) > len(feature_names):
            shap_vals = shap_vals[: len(feature_names)]

        # Step 5: Render Clinical Attribution Bar Plot
        self._plot_attributions(
            shap_vals=shap_vals,
            feature_names=feature_names,
            query_values=query_clin.squeeze().detach().cpu().numpy(),
            output_path=output_path,
        )

        return shap_vals

    def _plot_attributions(
        self,
        shap_vals: np.ndarray,
        feature_names: List[str],
        query_values: np.ndarray,
        output_path: str,
    ) -> None:
        """Renders an IEEE peer-review grade horizontal SHAP attribution chart."""
        # Sort features by absolute attribution magnitude
        order = np.argsort(np.abs(shap_vals))
        sorted_names = [feature_names[i] for i in order]
        sorted_vals = shap_vals[order]
        sorted_inputs = query_values[order]

        # Crimson for positive risk drivers, Steel Blue for protective factors
        colors = ["#e63946" if val >= 0 else "#457b9d" for val in sorted_vals]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        bars = ax.barh(range(len(sorted_names)), sorted_vals, color=colors, height=0.65, edgecolor="none")

        # Styling
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=11, fontweight="bold")
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)

        # Annotate bar tips with numerical attributions and original patient values
        x_lim = max(float(abs(sorted_vals).max()) * 1.35, 0.05)
        ax.set_xlim(-x_lim, x_lim)

        for idx, (bar, val, inp) in enumerate(zip(bars, sorted_vals, sorted_inputs)):
            offset = 0.005 if val >= 0 else -0.005
            ha = "left" if val >= 0 else "right"
            val_text = f"{val:+.3f} (val: {inp:.2f})"
            ax.text(
                val + offset,
                idx,
                val_text,
                va="center",
                ha=ha,
                fontsize=9.5,
                fontweight="semibold",
                color="#1d3557",
            )

        ax.set_xlabel(
            "SHAP Value (Marginal Impact on PAM Diagnostic Risk Probability)",
            fontsize=11,
            fontweight="bold",
            labelpad=10,
        )
        ax.set_title(
            "Multimodal Clinical Feature Attributions (SHAP DeepExplainer)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.5)

        # Legend handles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e63946", label="Increases PAM Risk (+)"),
            Patch(facecolor="#457b9d", label="Protective Factor (-)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10)

        plt.tight_layout()
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_file), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SHAP clinical attribution chart successfully saved to: {out_file.resolve()}")
