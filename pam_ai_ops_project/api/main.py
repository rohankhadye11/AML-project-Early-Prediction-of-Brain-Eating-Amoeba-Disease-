"""FastAPI Production Server for Multimodal PAM Clinical Decision Support System.

Module: api/main.py
Exposes high-performance RESTful inference and Explainable AI (XAI) endpoints:
- POST /api/predict_multimodal: Accepts uploaded brain MRI scan + 9 bedside clinical parameters,
  computes PAM risk probability via PAMLateFusionNet, and returns Grad-CAM visual heatmaps (base64)
  alongside SHAP feature attribution weights.
"""

from typing import Any, Dict, List, Optional, Tuple
import base64
import json
import logging
from pathlib import Path
import sys

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field
import torch
import torch.nn as nn

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in [PROJECT_ROOT, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clinical_shap import ClinicalSHAPExplainer
from multimodal_model import PAMLateFusionNet
from vision_gradcam import MRIExplainer

logger = logging.getLogger("api.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

# Standard 9 RFE-Optimized Clinical Features
CLINICAL_FEATURE_NAMES: List[str] = [
    "Freshwater_Exposure",
    "Severe_Headache",
    "Fever",
    "Nausea_Vomiting",
    "Stiff_Neck",
    "Photophobia",
    "Age",
    "Symptom_Onset_Days",
    "CSF_Pressure",
]


# =============================================================================
# Pydantic Schemas
# =============================================================================

class MultimodalPredictionResponse(BaseModel):
    """Schema for multimodal prediction with XAI attributions."""

    risk_probability: float = Field(
        ...,
        description="Predicted probability of Primary Amoebic Meningoencephalitis [0.0 - 1.0].",
        ge=0.0,
        le=1.0,
    )
    shap_values: Dict[str, float] = Field(
        ...,
        description="Shapley attribution weights for each of the 9 clinical features.",
    )
    gradcam_base64: str = Field(
        ...,
        description="Base64-encoded PNG string of the Grad-CAM neuroimaging heatmap overlay.",
    )
    triage_level: str = Field(
        ...,
        description="Clinical triage classification ('CRITICAL_HIGH_RISK' or 'LOW_RISK').",
    )
    status: str = Field(default="Success", description="Inference execution status.")


# =============================================================================
# Application Initialization & Lifespan / Model Cache
# =============================================================================

app = FastAPI(
    title="PAM Multimodal Clinical Decision Support API",
    description=(
        "Production API serving the PyTorch Late Fusion Neural Network and "
        "Explainable AI (Grad-CAM & SHAP) engines for rapid triage of Naegleria fowleri."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Next.js Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Cached Artifacts
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[PAMLateFusionNet] = None
_background_clinical: Optional[torch.Tensor] = None


def get_model() -> PAMLateFusionNet:
    """Retrieves or lazily instantiates the singleton PAMLateFusionNet model."""
    global _model
    if _model is not None:
        return _model

    logger.info("Initializing PAMLateFusionNet instance...")
    model = PAMLateFusionNet(
        num_clinical_features=9,
        image_embedding_dim=64,
        clinical_embedding_dim=32,
        pretrained=False,
    )

    # Search for pre-trained weights
    search_paths = [
        PROJECT_ROOT / "best_multimodal_model.pth",
        PROJECT_ROOT / "models" / "best_multimodal_model.pth",
        Path("best_multimodal_model.pth"),
    ]

    loaded = False
    for path in search_paths:
        if path.exists():
            try:
                state_dict = torch.load(str(path), map_location=_device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded trained weights from: {path.resolve()}")
                loaded = True
                break
            except Exception as exc:
                logger.warning(f"Failed loading checkpoint from {path}: {exc}")

    if not loaded:
        logger.warning("No checkpoint found; running with initialized model weights.")

    if not hasattr(model, "vision_branch"):
        model.vision_branch = model.image_backbone

    model.to(_device)
    model.eval()
    _model = model
    return _model


def get_background_clinical() -> torch.Tensor:
    """Retrieves or synthesizes the reference clinical cohort for SHAP DeepExplainer."""
    global _background_clinical
    if _background_clinical is not None:
        return _background_clinical

    rng = np.random.default_rng(seed=42)
    bg_samples = np.column_stack([
        rng.binomial(1, 0.25, size=100),  # Freshwater
        rng.binomial(1, 0.45, size=100),  # Headache
        rng.binomial(1, 0.50, size=100),  # Fever
        rng.binomial(1, 0.35, size=100),  # Nausea
        rng.binomial(1, 0.20, size=100),  # Stiff neck
        rng.binomial(1, 0.20, size=100),  # Photophobia
        rng.uniform(0.1, 0.8, size=100),  # Age
        rng.uniform(0.2, 0.8, size=100),  # Onset days
        rng.uniform(0.1, 0.6, size=100),  # CSF Pressure
    ]).astype(np.float32)

    _background_clinical = torch.from_numpy(bg_samples).to(_device)
    return _background_clinical


# =============================================================================
# Helper Utilities
# =============================================================================

def parse_and_validate_clinical_json(clinical_data_str: str) -> np.ndarray:
    """Parses JSON string and formats into a 9-dimensional float32 vector.

    Args:
        clinical_data_str: JSON string containing the clinical features.

    Returns:
        np.ndarray: Array of shape (1, 9) in float32.

    Raises:
        HTTPException: If JSON is invalid or missing critical features.
    """
    try:
        data_dict = json.loads(clinical_data_str)
    except json.JSONDecodeError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid clinical_data JSON payload: {err.msg}",
        )

    vector = []
    for feat in CLINICAL_FEATURE_NAMES:
        if feat in data_dict:
            try:
                vector.append(float(data_dict[feat]))
            except (ValueError, TypeError):
                vector.append(0.0)
        else:
            # Check lowercase / normalized alias
            alias_match = next(
                (v for k, v in data_dict.items() if k.lower().replace(" ", "_") == feat.lower()),
                None,
            )
            if alias_match is not None:
                vector.append(float(alias_match))
            else:
                vector.append(0.0)

    return np.array([vector], dtype=np.float32)


def process_uploaded_image(file_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
    """Decodes uploaded image bytes, converts to grayscale 224x224, and produces tensors.

    Args:
        file_bytes: Raw binary image payload.

    Returns:
        Tuple[torch.Tensor, np.ndarray]:
            - image_tensor: Float tensor of shape (1, 1, 224, 224) normalized to [0, 1].
            - original_image_array: Grayscale uint8 array of shape (224, 224).

    Raises:
        HTTPException: If image cannot be decoded.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        # Fallback to grayscale read
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to decode uploaded image. Supported formats: PNG, JPEG, DICOM slices.",
            )
    else:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Standardize to 224x224 spatial resolution
    resized_gray = cv2.resize(img_gray, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0] float tensor
    norm_tensor = torch.from_numpy(resized_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return norm_tensor, resized_gray


# =============================================================================
# Primary Multimodal Prediction & XAI Endpoint
# =============================================================================

@app.post(
    "/api/predict_multimodal",
    response_model=MultimodalPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference & Explainability"],
    summary="Multimodal Late Fusion Prediction + Grad-CAM & SHAP Explanation",
)
async def predict_multimodal(
    image: Optional[UploadFile] = File(None, description="Uploaded brain MRI slice (224x224 or resizable)"),
    file: Optional[UploadFile] = File(None, description="Alternative parameter alias for uploaded brain MRI"),
    clinical_data: str = Form(
        ...,
        description="JSON string encoding the clinical features (e.g., {'Freshwater_Exposure': 1, ...})",
    ),
):
    """
    Executes end-to-end multimodal inference:
    1. Preprocesses uploaded neuroimaging scan and bedside clinical vector.
    2. Passes dual inputs through PAMLateFusionNet to compute risk probability.
    3. Runs MRIExplainer (Grad-CAM) to identify neuropathological foci and returns a base64 PNG.
    4. Runs ClinicalSHAPExplainer (SHAP) to attribute feature contributions across the 9 parameters.
    """
    try:
        # Step 1: Decode Inputs
        target_file = image or file
        if target_file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image uploaded. Please provide an MRI file under 'image' or 'file'.",
            )
        contents = await target_file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty.",
            )

        image_tensor, original_gray = process_uploaded_image(contents)
        clinical_array = parse_and_validate_clinical_json(clinical_data)
        clinical_tensor = torch.from_numpy(clinical_array)

        # Step 2: Model Inference
        model = get_model()
        image_tensor = image_tensor.to(_device)
        clinical_tensor = clinical_tensor.to(_device)

        with torch.no_grad():
            raw_logit = model(image_tensor, clinical_tensor)
            risk_probability = float(torch.sigmoid(raw_logit).item())

        triage_level = "CRITICAL_HIGH_RISK" if risk_probability >= 0.50 else "LOW_RISK"

        # Step 3: Trigger Vision Explainer (Grad-CAM)
        mri_explainer = MRIExplainer(model=model, use_cuda=(_device.type == "cuda"))
        cam_visualization = mri_explainer.generate_heatmap(
            image_tensor=image_tensor,
            clinical_tensor=clinical_tensor,
            original_image_array=original_gray,
            output_path=str(PROJECT_ROOT / "scratch_gradcam.png"),
        )

        # Convert RGB visualization to Base64 PNG string
        bgr_vis = cv2.cvtColor(cam_visualization, cv2.COLOR_RGB2BGR)
        success, encoded_png = cv2.imencode(".png", bgr_vis)
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM overlay to PNG.")
        base64_str = base64.b64encode(encoded_png.tobytes()).decode("utf-8")
        gradcam_base64 = f"data:image/png;base64,{base64_str}"

        # Step 4: Trigger Tabular Explainer (SHAP)
        bg_clinical = get_background_clinical()
        shap_explainer = ClinicalSHAPExplainer(
            model=model,
            background_clinical=bg_clinical,
            background_image=image_tensor,
        )

        shap_weights = shap_explainer.explain_patient(
            patient_clinical_tensor=clinical_tensor,
            feature_names=CLINICAL_FEATURE_NAMES,
            patient_image_tensor=image_tensor,
            output_path=str(PROJECT_ROOT / "scratch_shap.png"),
        )

        shap_dict = {
            feat_name: float(weight)
            for feat_name, weight in zip(CLINICAL_FEATURE_NAMES, shap_weights)
        }

        logger.info(
            f"Successfully processed patient prediction: Risk={risk_probability:.4f} ({triage_level})"
        )

        return MultimodalPredictionResponse(
            risk_probability=round(risk_probability, 4),
            shap_values=shap_dict,
            gradcam_base64=gradcam_base64,
            triage_level=triage_level,
            status="Success",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Prediction pipeline encountered error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint confirming operational readiness."""
    return {
        "status": "Healthy",
        "device": str(_device),
        "model_loaded": _model is not None,
        "features": CLINICAL_FEATURE_NAMES,
    }
