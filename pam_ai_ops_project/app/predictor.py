"""
Core ML Inference Engine and Image Feature Extraction for PAM CDS System.
Module: app/predictor.py
"""

from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd


class PAMPredictor:
    """
    Production predictor class managing ML inference pipelines and
    multimodal neuroimaging feature extraction for Primary Amoebic Meningoencephalitis (PAM).
    """

    NUMERIC_COLUMNS = ["Age", "Symptom_Onset_Days", "CSF_Pressure", "CSF_WBC_Count"]

    ALL_RAW_FEATURES = [
        "Freshwater_Exposure",
        "Nasal_Irrigation_History",
        "Severe_Headache",
        "Fever",
        "Nausea_Vomiting",
        "Stiff_Neck",
        "Confusion",
        "Seizures",
        "Photophobia",
        "Age",
        "Symptom_Onset_Days",
        "CSF_Pressure",
        "CSF_WBC_Count",
    ]

    def __init__(self, artifacts_dir: Path | str | None = None):
        """
        Load persisted Phase 1 and Phase 2 artifacts:
        - scaler.pkl: Fitted MinMaxScaler for continuous clinical vitals
        - selected_features.pkl: 9 RFE-optimized feature names
        - best_model.pkl: Trained top-performing Random Forest Classifier
        """
        project_root = Path(__file__).resolve().parent.parent
        if artifacts_dir is None:
            self.artifacts_dir = project_root / "models" / "artifacts"
        else:
            self.artifacts_dir = Path(artifacts_dir)

        self.scaler_path = self.artifacts_dir / "scaler.pkl"
        self.features_path = self.artifacts_dir / "selected_features.pkl"
        self.model_path = self.artifacts_dir / "best_model.pkl"
        self.multimodal_model_path = self.artifacts_dir / "multimodal_model.pkl"

        self.scaler = None
        self.selected_features = None
        self.model = None
        self.multimodal_model = None
        self.is_ready = False
        self.has_multimodal = False
        self.init_error = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Internal helper to load model artifacts safely."""
        try:
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler artifact not found at {self.scaler_path}")
            if not self.features_path.exists():
                raise FileNotFoundError(f"Selected features artifact not found at {self.features_path}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"Trained model artifact not found at {self.model_path}")

            self.scaler = joblib.load(self.scaler_path)
            self.selected_features = joblib.load(self.features_path)
            self.model = joblib.load(self.model_path)
            self.is_ready = True
            print(f"[PAMPredictor] Core artifacts loaded successfully from {self.artifacts_dir}")
            print(f"[PAMPredictor] Selected 9 features: {self.selected_features}")
            
            # Load multimodal fusion model if present
            if self.multimodal_model_path.exists():
                self.multimodal_model = joblib.load(self.multimodal_model_path)
                self.has_multimodal = True
                print(f"[PAMPredictor] Multimodal fusion model loaded from {self.multimodal_model_path}")
            else:
                print(f"[PAMPredictor WARNING] Multimodal model not found at {self.multimodal_model_path}")
        except Exception as exc:
            self.is_ready = False
            self.init_error = str(exc)
            print(f"[PAMPredictor ERROR] Failed to load artifacts: {exc}")

    def predict_symptoms(self, data: dict) -> dict:
        """
        Generate clinical risk prediction from raw 13 patient symptoms/biomarkers.

        Parameters:
        -----------
        data : dict
            Dictionary containing the 13 raw clinical features.

        Returns:
        --------
        dict:
            {
                "prediction": int (0 for Negative, 1 for PAM Positive),
                "risk_probability": float (0.0 to 1.0),
                "status": "Success"
            }
        """
        if not self.is_ready:
            raise RuntimeError(f"Predictor artifacts not loaded: {self.init_error}")

        # 1. Convert input to DataFrame
        df_input = pd.DataFrame([data])

        # Validate feature completeness
        missing = [col for col in self.ALL_RAW_FEATURES if col not in df_input.columns]
        if missing:
            raise ValueError(f"Missing required clinical features: {missing}")

        # 2. Normalize numeric features using fitted MinMaxScaler
        df_input[self.NUMERIC_COLUMNS] = self.scaler.transform(df_input[self.NUMERIC_COLUMNS])

        # 3. Filter down to the 9 features selected by RFE
        df_opt = df_input[self.selected_features]

        # 4. Generate binary prediction and probability
        prediction = int(self.model.predict(df_opt)[0])
        probabilities = self.model.predict_proba(df_opt)[0]
        risk_probability = float(probabilities[1])  # Probability of PAM (class 1)

        return {
            "prediction": prediction,
            "risk_probability": round(risk_probability, 4),
            "status": "Success",
        }

    def extract_mri_features(self, image_bytes: bytes) -> dict:
        """
        Process neuroimaging (MRI) scan bytes and extract a baseline
        50-dimensional feature vector simulating multimodal neuroimaging integration.

        Parameters:
        -----------
        image_bytes : bytes
            Raw bytes of the uploaded MRI scan.

        Returns:
        --------
        dict:
            {
                "feature_vector": list of 50 floats,
                "dimensions": 50,
                "status": "Success",
                "message": str
            }
        """
        # 1. Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Unable to decode uploaded image file. Ensure it is a valid JPEG, PNG, or DICOM-compatible format.")

        # 2. Resize to 224x224 pixels
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # 3. Convert to Grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # 4. Extract 50-dimensional feature vector:
        # Part A: 32-bin normalized intensity histogram
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() + 1e-7)  # 32 features

        # Part B: 4 Quadrant mean intensities and 4 quadrant standard deviations (8 features)
        h, w = gray.shape
        half_h, half_w = h // 2, w // 2
        q1 = gray[:half_h, :half_w]
        q2 = gray[:half_h, half_w:]
        q3 = gray[half_h:, :half_w]
        q4 = gray[half_h:, half_w:]

        quad_means = [float(q.mean() / 255.0) for q in [q1, q2, q3, q4]]  # 4 features
        quad_stds = [float(q.std() / 255.0) for q in [q1, q2, q3, q4]]    # 4 features

        # Part C: Global statistical metrics and central ROI metrics (10 features)
        center_roi = gray[half_h // 2 : half_h + half_h // 2, half_w // 2 : half_w + half_w // 2]
        global_stats = [
            float(gray.mean() / 255.0),
            float(gray.std() / 255.0),
            float(np.median(gray) / 255.0),
            float(gray.min() / 255.0),
            float(gray.max() / 255.0),
            float(np.percentile(gray, 10) / 255.0),
            float(np.percentile(gray, 25) / 255.0),
            float(np.percentile(gray, 75) / 255.0),
            float(np.percentile(gray, 90) / 255.0),
            float(center_roi.mean() / 255.0),
        ]  # 10 features

        # Combine: 32 + 4 + 4 + 10 = 50 features
        feature_vector_50 = np.concatenate([hist_norm, quad_means, quad_stds, global_stats])
        feature_vector_50 = [round(float(val), 6) for val in feature_vector_50]

        return {
            "feature_vector": feature_vector_50,
            "dimensions": len(feature_vector_50),
            "status": "Success",
            "message": "Extracted 50-dimensional neuroimaging feature representation successfully.",
        }

    def predict_multimodal(self, clinical_data: dict, image_bytes: bytes) -> dict:
        """
        Unified multimodal prediction method fusing 9 scaled clinical features
        with 50 extracted neuroimaging features into a 59-dimensional vector.

        Parameters:
        -----------
        clinical_data : dict
            Dictionary of the 13 raw clinical symptoms and biomarkers.
        image_bytes : bytes
            Raw byte stream of the brain MRI/CT scan.

        Returns:
        --------
        dict:
            {
                "prediction": int (0 for Negative, 1 for PAM Positive),
                "risk_probability": float (0.0 to 1.0),
                "status": "Success",
                "dimensions": 59,
                "clinical_features_used": 9,
                "mri_features_used": 50,
            }
        """
        if not self.is_ready:
            raise RuntimeError(f"Predictor artifacts not loaded: {self.init_error}")
        if not self.has_multimodal or self.multimodal_model is None:
            raise RuntimeError("Multimodal fusion model artifact (multimodal_model.pkl) is not loaded.")

        # 1. Process clinical features (scale numeric, filter to 9 RFE features)
        df_input = pd.DataFrame([clinical_data])
        missing = [col for col in self.ALL_RAW_FEATURES if col not in df_input.columns]
        if missing:
            raise ValueError(f"Missing required clinical features: {missing}")

        df_input[self.NUMERIC_COLUMNS] = self.scaler.transform(df_input[self.NUMERIC_COLUMNS])
        df_clinical_opt = df_input[self.selected_features].copy()

        # 2. Process image bytes into 50-D feature vector
        mri_dict = self.extract_mri_features(image_bytes)
        mri_vector = mri_dict["feature_vector"]

        # 3. Concatenate arrays: [9 clinical features, 50 MRI features]
        mri_cols = [f"mri_feature_{idx:02d}" for idx in range(50)]
        df_mri = pd.DataFrame([mri_vector], columns=mri_cols)

        df_fused = pd.concat([df_clinical_opt.reset_index(drop=True), df_mri.reset_index(drop=True)], axis=1)

        # 4. Multimodal Model Inference
        prediction = int(self.multimodal_model.predict(df_fused)[0])
        probabilities = self.multimodal_model.predict_proba(df_fused)[0]
        risk_probability = float(probabilities[1])

        return {
            "prediction": prediction,
            "risk_probability": round(risk_probability, 4),
            "status": "Success",
            "dimensions": int(df_fused.shape[1]),
            "clinical_features_used": len(self.selected_features),
            "mri_features_used": len(mri_vector),
        }

