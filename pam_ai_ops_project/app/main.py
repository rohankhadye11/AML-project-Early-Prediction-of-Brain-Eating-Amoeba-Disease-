"""
FastAPI Real-Time Web Deployment for Primary Amoebic Meningoencephalitis (PAM) CDS System.
Module: app/main.py

Provides:
- GET /: API metadata and endpoints overview
- GET /health: Health check and artifact readiness
- POST /predict/clinical: "Symptoms Only" clinical ML risk stratification
- POST /predict/multimodal: "Symptoms + MRI" multimodal neuroimaging feature extraction
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import PAMPredictor

# =============================================================================
# 1. APPLICATION INITIALIZATION & GLOBAL PREDICTOR INSTANTIATION
# =============================================================================
app = FastAPI(
    title="PAM Early Prediction API",
    description=(
        "Production Clinical Decision Support System for early risk stratification "
        "of Primary Amoebic Meningoencephalitis (PAM) caused by Naegleria fowleri."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Next.js frontend integration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance with Graceful Error Handling
try:
    predictor = PAMPredictor()
except Exception as exc:
    print(f"[CRITICAL ERROR] Failed to instantiate PAMPredictor: {exc}")
    predictor = None

# Attach Phase 5/6 Multimodal Late Fusion & XAI Router (Grad-CAM & SHAP)
try:
    from api.main import app as xai_app
    app.include_router(xai_app.router)
    print("[PAM] Attached Phase 6 Multimodal & XAI Router (/api/predict_multimodal) to main API.")
except Exception as exc:
    print(f"[PAM Warning] Failed to attach XAI router: {exc}")



def check_predictor_readiness():
    """Helper to verify model artifacts are loaded; raises HTTP 500 if unready."""
    if predictor is None or not predictor.is_ready:
        error_msg = predictor.init_error if predictor else "Predictor failed to initialize"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical decision model artifacts unavailable: {error_msg}. Verify models/artifacts/ directory.",
        )


# =============================================================================
# 2. PYDANTIC SCHEMAS
# =============================================================================
class ClinicalInput(BaseModel):
    """Schema for the 13 raw clinical symptoms and biomarkers."""

    Freshwater_Exposure: int = Field(
        ...,
        description="History of swimming/diving in warm freshwater (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Nasal_Irrigation_History: int = Field(
        ...,
        description="History of unboiled tap water nasal/sinus irrigation or ritual ablution (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Severe_Headache: int = Field(
        ...,
        description="Acute excruciating bifrontal or bitemporal headache (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Fever: int = Field(
        ...,
        description="High pyrexia / fever (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Nausea_Vomiting: int = Field(
        ...,
        description="Nausea and projectile vomiting indicating elevated ICP (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Stiff_Neck: int = Field(
        ...,
        description="Nuchal rigidity / meningismus (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Confusion: int = Field(
        ...,
        description="Altered mental status, delirium, or lethargy (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Seizures: int = Field(
        ...,
        description="Generalized or focal seizure activity (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Photophobia: int = Field(
        ...,
        description="Abnormal visual sensitivity to light (1 = Yes, 0 = No)",
        ge=0,
        le=1,
    )
    Age: float = Field(
        ...,
        description="Patient age in years (1 to 100)",
        ge=1.0,
        le=100.0,
    )
    Symptom_Onset_Days: float = Field(
        ...,
        description="Days elapsed since initial clinical symptom onset (1 to 30)",
        ge=0.0,
        le=30.0,
    )
    CSF_Pressure: float = Field(
        ...,
        description="Lumbar puncture opening CSF pressure in mmH2O (typically 100 - 500+)",
        ge=50.0,
        le=600.0,
    )
    CSF_WBC_Count: float = Field(
        ...,
        description="Cerebrospinal fluid leukocyte pleocytosis count in cells/mm3 (0 - 1500+)",
        ge=0.0,
        le=2000.0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Freshwater_Exposure": 1,
                    "Nasal_Irrigation_History": 0,
                    "Severe_Headache": 1,
                    "Fever": 1,
                    "Nausea_Vomiting": 1,
                    "Stiff_Neck": 1,
                    "Confusion": 1,
                    "Seizures": 0,
                    "Photophobia": 1,
                    "Age": 21.0,
                    "Symptom_Onset_Days": 4.0,
                    "CSF_Pressure": 380.0,
                    "CSF_WBC_Count": 610.0,
                }
            ]
        }
    }


class ClinicalPredictionResponse(BaseModel):
    """Standardized response schema for clinical symptom inference."""

    prediction: int = Field(
        ...,
        description="0 = Negative for PAM (Acute Non-PAM Meningitis), 1 = Positive for PAM",
    )
    risk_probability: float = Field(
        ...,
        description="Estimated posterior probability of PAM infection (0.0 to 1.0)",
    )
    status: str = Field(
        default="Success",
        description="API execution status",
    )
    clinical_interpretation: Optional[str] = Field(
        default=None,
        description="Clinical triage guidance based on estimated risk",
    )


class MultimodalPredictionResponse(BaseModel):
    """Standardized response schema for multimodal clinical + neuroimaging prediction."""

    prediction: int = Field(
        ...,
        description="0 = Negative for PAM, 1 = Positive for PAM",
    )
    risk_probability: float = Field(
        ...,
        description="Estimated posterior probability of PAM infection from fused multimodal model",
    )
    status: str = Field(
        default="Success",
        description="API execution status",
    )
    clinical_interpretation: Optional[str] = Field(
        default=None,
        description="Clinical triage guidance based on estimated multimodal risk",
    )
    multimodal_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata on fused clinical and neuroimaging feature dimensions",
    )



# =============================================================================
# 3. FASTAPI ROUTING & ENDPOINTS
# =============================================================================
@app.get("/", tags=["System"])
def root():
    """Root metadata endpoint with system status and documentation links."""
    return {
        "title": "PAM Early Prediction API",
        "description": "Clinical Decision Support System for Primary Amoebic Meningoencephalitis",
        "version": "1.0.0",
        "status": "Online" if predictor and predictor.is_ready else "Degraded",
        "docs_url": "/docs",
        "endpoints": {
            "clinical_prediction": "POST /predict/clinical",
            "multimodal_prediction": "POST /predict/multimodal",
            "health_check": "GET /health",
            "selected_features": "GET /features",
        },
    }


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint to verify backend operational readiness and model artifact loading."""
    check_predictor_readiness()
    return {
        "status": "Healthy",
        "predictor_ready": True,
        "selected_features_count": len(predictor.selected_features),
        "model_type": type(predictor.model).__name__,
        "artifacts_directory": str(predictor.artifacts_dir),
    }


@app.get("/features", tags=["System"])
def get_selected_features():
    """Returns the list of 13 raw input features and 9 RFE-optimized features."""
    check_predictor_readiness()
    return {
        "all_raw_features": predictor.ALL_RAW_FEATURES,
        "numeric_features": predictor.NUMERIC_COLUMNS,
        "selected_features_rfe_9": predictor.selected_features,
    }


@app.post(
    "/predict/clinical",
    response_model=ClinicalPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Symptoms Only Mode",
)
def predict_clinical(payload: ClinicalInput):
    """
    Endpoint 1: 'Symptoms Only' Mode
    Accepts 13 patient symptoms/biomarkers, normalizes numeric parameters,
    filters to 9 optimal features, and generates clinical risk stratification.
    """
    check_predictor_readiness()

    try:
        input_dict = payload.model_dump()
        result = predictor.predict_symptoms(input_dict)

        # Add clinical guidance interpretation
        prob = result["risk_probability"]
        if prob >= 0.70:
            guidance = (
                "CRITICAL HIGH RISK: Strongly suggestive of PAM. Immediate confirmatory CSF wet-mount/PCR "
                "and empirical treatment (miltefosine, liposomal amphotericin B, hypothermia) recommended."
            )
        elif prob >= 0.40:
            guidance = (
                "MODERATE RISK: Equivocal presentation. Urgent neurological consultation, "
                "close ICP monitoring, and amoebic PCR screening advised."
            )
        else:
            guidance = (
                "LOW RISK: Presentation more consistent with typical acute bacterial/viral meningitis. "
                "Maintain standard antimicrobial protocols and observe for acute deterioration."
            )

        return ClinicalPredictionResponse(
            prediction=result["prediction"],
            risk_probability=result["risk_probability"],
            status="Success",
            clinical_interpretation=guidance,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Clinical inference error: {str(exc)}",
        )


@app.post(
    "/predict/multimodal",
    response_model=MultimodalPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Symptoms + MRI Mode",
)
async def predict_multimodal(
    file: UploadFile = File(..., description="Neuroimaging (Brain MRI/CT) scan file (JPEG/PNG/DICOM)"),
    clinical_data: str = Form(
        ...,
        description=(
            "JSON-encoded string containing the 13 raw clinical features. "
            "Example: {\"Freshwater_Exposure\": 1, \"Nasal_Irrigation_History\": 0, ...}"
        ),
    ),
):
    """
    Endpoint 2: 'Symptoms + MRI' Mode
    Accepts both a neuroimaging file and a JSON-encoded string of 13 clinical features.
    Extracts 50-D MRI features, normalizes clinical features to 9 RFE features,
    fuses into a 59-dimensional array, and executes the trained Multimodal Random Forest model.
    """
    check_predictor_readiness()

    # 1. Parse clinical data JSON string
    try:
        clinical_dict = json.loads(clinical_data)
    except json.JSONDecodeError as json_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid clinical_data JSON string: {str(json_err)}",
        )

    # Validate against ClinicalInput schema
    try:
        validated_input = ClinicalInput(**clinical_dict)
    except Exception as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Clinical input validation error: {str(val_err)}",
        )

    # 2. Read image file bytes
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file uploaded. Please upload a valid MRI scan.",
            )

        # 3. Execute Multimodal Prediction
        result = predictor.predict_multimodal(validated_input.model_dump(), image_bytes)

        # 4. Generate Clinical Interpretation
        prob = result["risk_probability"]
        if prob >= 0.70:
            guidance = (
                "CRITICAL HIGH RISK (MULTIMODAL FUSION): Strong concordance between clinical symptom "
                "presentation and neuroimaging findings indicating cerebral edema/necrotizing lesions. "
                "Emergency confirmatory CSF PCR and intensive intervention (miltefosine, liposomal amphotericin B, "
                "hypothermia) immediately indicated."
            )
        elif prob >= 0.40:
            guidance = (
                "MODERATE RISK (MULTIMODAL FUSION): Equivocal neuroimaging findings with elevated risk profile. "
                "Immediate neurological ICU admission, intracranial pressure monitoring, and rapid PCR testing advised."
            )
        else:
            guidance = (
                "LOW RISK (MULTIMODAL FUSION): Clinical biomarkers and neuroimaging analysis do not demonstrate acute "
                "meningoencephalitic edema characteristic of early PAM. Continue non-amoebic meningitis differential workup."
            )

        return MultimodalPredictionResponse(
            prediction=result["prediction"],
            risk_probability=result["risk_probability"],
            status="Success",
            clinical_interpretation=guidance,
            multimodal_details={
                "filename": file.filename or "uploaded_mri.png",
                "total_fused_features": result["dimensions"],
                "clinical_features_count": result["clinical_features_used"],
                "mri_features_count": result["mri_features_used"],
            },
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image processing error: {str(val_err)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multimodal prediction error: {str(exc)}",
        )


# =============================================================================
# 4. EXECUTION INSTRUCTIONS
# =============================================================================
# To launch the FastAPI server locally or in production:
#
# Terminal Command:
#   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
# Interactive Swagger Documentation:
#   http://localhost:8000/docs
#
# ReDoc Alternative Documentation:
#   http://localhost:8000/redoc
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
