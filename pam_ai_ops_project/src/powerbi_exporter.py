"""
Power BI Data Aggregation and Export Script.
Module: src/powerbi_exporter.py

Generates a denormalized, star-schema ready analytical dataset combining
clinical biomarkers, neuroimaging feature predictions, and classification diagnostics.
Exports to: data/processed/powerbi_multimodal_analytics.csv
"""

from pathlib import Path
import uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Feature definitions
CLINICAL_9_FEATURES = [
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


def simulate_mri_features(
    y: pd.Series,
    random_state: int = 142,
    shift_strength: float = 0.12,
    p_detect: float = 0.85,
) -> pd.DataFrame:
    """
    Generate 50 simulated continuous neuroimaging features consistent with Phase 4 pipeline.
    """
    np.random.seed(random_state)
    n_samples = len(y)
    mri_matrix = np.random.normal(loc=0.15, scale=0.06, size=(n_samples, 50))
    edema_indices = [10, 15, 20, 32, 33, 40, 48, 49]

    for i in range(n_samples):
        label = y.iloc[i]
        if label == 1 and np.random.rand() < p_detect:
            shift_noise = np.random.normal(shift_strength, 0.02, size=len(edema_indices))
            mri_matrix[i, edema_indices] += shift_noise

    mri_matrix = np.clip(mri_matrix, 0.0, 1.0)
    cols = [f"mri_feature_{idx:02d}" for idx in range(50)]
    return pd.DataFrame(mri_matrix, columns=cols)


def generate_powerbi_dataset(
    processed_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    output_filename: str = "powerbi_multimodal_analytics.csv",
) -> pd.DataFrame:
    """
    Generate denormalized tabular analytics dataset for Power BI reporting.

    Parameters:
    -----------
    processed_dir : Path or str, optional
        Path to processed data directory.
    artifacts_dir : Path or str, optional
        Path to model artifacts directory.
    output_filename : str
        Target CSV filename in processed_dir.

    Returns:
    --------
    pd.DataFrame
        Analytical cohort DataFrame ready for Power BI.
    """
    project_root = Path(__file__).resolve().parent.parent
    if processed_dir is None:
        processed_dir = project_root / "data" / "processed"
    else:
        processed_dir = Path(processed_dir)

    if artifacts_dir is None:
        artifacts_dir = project_root / "models" / "artifacts"
    else:
        artifacts_dir = Path(artifacts_dir)

    # 1. Load Artifacts & Test Cohort
    x_test_path = processed_dir / "X_test_opt.csv"
    y_test_path = processed_dir / "y_test.csv"
    model_path = artifacts_dir / "multimodal_model.pkl"

    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(f"Test split datasets missing from {processed_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Multimodal model artifact missing from {model_path}")

    X_test_opt = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    multimodal_model = joblib.load(model_path)

    print(f"Loaded clinical test cohort: {X_test_opt.shape}")
    print(f"Loaded multimodal model: {type(multimodal_model).__name__}")

    # 2. Simulate Matching 50-D MRI Features
    mri_test = simulate_mri_features(y_test, random_state=142)

    # 3. Concatenate into 59-D Multimodal Input
    X_test_multi = pd.concat([X_test_opt, mri_test], axis=1)

    # 4. Generate Predictions and Confidence Probabilities
    y_pred = multimodal_model.predict(X_test_multi)
    y_prob = multimodal_model.predict_proba(X_test_multi)[:, 1]

    # 5. Construct Denormalized Power BI Table
    df_powerbi = X_test_opt.copy()

    # Generate standardized clinical Patient IDs
    patient_ids = [f"PAM-PT-{idx + 1:04d}" for idx in range(len(df_powerbi))]
    df_powerbi.insert(0, "Patient_ID", patient_ids)

    # Append Target, Predictions, and Confidence
    df_powerbi["Actual_Outcome"] = y_test.values
    df_powerbi["Predicted_Outcome"] = y_pred
    df_powerbi["Risk_Probability"] = np.round(y_prob, 4)

    # Compute Diagnostic Prediction Type (Confusion Matrix Category)
    prediction_types = []
    for actual, pred in zip(y_test.values, y_pred):
        if actual == 1 and pred == 1:
            prediction_types.append("True Positive")
        elif actual == 0 and pred == 0:
            prediction_types.append("True Negative")
        elif actual == 0 and pred == 1:
            prediction_types.append("False Positive")
        elif actual == 1 and pred == 0:
            prediction_types.append("False Negative")
        else:
            prediction_types.append("Unknown")

    df_powerbi["Prediction_Type"] = prediction_types

    # Add Power BI Analytical Helper Columns (Slicers / DAX Measures)
    df_powerbi["Risk_Tier"] = pd.cut(
        df_powerbi["Risk_Probability"],
        bins=[-0.01, 0.3999, 0.6999, 1.0],
        labels=["Low Risk (<40%)", "Moderate Risk (40-69%)", "Critical High Risk (>=70%)"],
    )
    df_powerbi["Actual_Label"] = df_powerbi["Actual_Outcome"].map({1: "PAM Positive", 0: "Negative Control"})
    df_powerbi["Predicted_Label"] = df_powerbi["Predicted_Outcome"].map({1: "PAM Positive", 0: "Negative Control"})

    # 6. Save Flat CSV for Direct Star Schema Ingestion
    out_path = processed_dir / output_filename
    df_powerbi.to_csv(out_path, index=False)
    print(f"\nPower BI analytical dataset successfully exported to: {out_path}")
    print(f"Shape: {df_powerbi.shape}")
    print(f"Columns ({len(df_powerbi.columns)}): {df_powerbi.columns.tolist()}")

    # 7. Print Diagnostic Summary
    print("\n--- DIAGNOSTIC CLASSIFICATION BREAKDOWN ---")
    type_counts = df_powerbi["Prediction_Type"].value_counts()
    for cat, count in type_counts.items():
        pct = (count / len(df_powerbi)) * 100
        print(f"  {cat:16}: {count:4d} ({pct:5.2f}%)")

    return df_powerbi


def main():
    print("=" * 70)
    print("PHASE 5 - STEP 1: POWER BI DATA AGGREGATION & EXPORT")
    print("=" * 70)
    df_pbi = generate_powerbi_dataset()
    print("=" * 70)
    print("SAMPLE POWER BI EXPORT RECORDS (FIRST 3 ROWS):")
    display_cols = [
        "Patient_ID",
        "Freshwater_Exposure",
        "Stiff_Neck",
        "CSF_Pressure",
        "Actual_Outcome",
        "Predicted_Outcome",
        "Risk_Probability",
        "Prediction_Type",
    ]
    print(df_pbi[display_cols].head(3).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
