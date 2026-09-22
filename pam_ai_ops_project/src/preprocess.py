"""
Preprocessing Pipeline for PAM Clinical Decision Support System.
Performs:
1. Imputation (median for numeric, mode for binary)
2. Normalization (MinMaxScaler on numeric features, saving scaler.pkl)
3. Class Balancing (SMOTE oversampling to 820:820 ratio -> 1640 total)
4. Stratified Data Splitting (80/20 train-test split)
5. Exporting processed train/test datasets to data/processed/
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Feature Group Definitions
BINARY_FEATURES = [
    "Freshwater_Exposure",
    "Nasal_Irrigation_History",
    "Severe_Headache",
    "Fever",
    "Nausea_Vomiting",
    "Stiff_Neck",
    "Confusion",
    "Seizures",
    "Photophobia",
]

NUMERIC_FEATURES = [
    "Age",
    "Symptom_Onset_Days",
    "CSF_Pressure",
    "CSF_WBC_Count",
]

TARGET_COLUMN = "Target"


def preprocess_pam_data(
    raw_data_path: Path | str | None = None,
    processed_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    random_state: int = 42,
):
    """
    Execute the full preprocessing pipeline on raw PAM clinical data.

    Returns:
    --------
    tuple:
        (X_train, X_test, y_train, y_test, scaler, df_raw, df_smote)
    """
    # 0. Resolve Paths
    project_root = Path(__file__).resolve().parent.parent
    if raw_data_path is None:
        raw_data_path = project_root / "data" / "raw" / "synthetic_pam_raw_1000.csv"
    else:
        raw_data_path = Path(raw_data_path)

    if processed_dir is None:
        processed_dir = project_root / "data" / "processed"
    else:
        processed_dir = Path(processed_dir)

    if artifacts_dir is None:
        artifacts_dir = project_root / "models" / "artifacts"
    else:
        artifacts_dir = Path(artifacts_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Data
    df_raw = pd.read_csv(raw_data_path)
    print(f"Loaded raw cohort from {raw_data_path} with shape: {df_raw.shape}")

    df_cleaned = df_raw.copy()

    # 2. Imputation: median for numeric, mode for binary
    for col in NUMERIC_FEATURES:
        if col in df_cleaned.columns:
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)

    for col in BINARY_FEATURES:
        if col in df_cleaned.columns:
            mode_val = df_cleaned[col].mode()[0]
            df_cleaned[col] = df_cleaned[col].fillna(mode_val)

    # 3. Normalization: MinMaxScaler on numeric features
    scaler = MinMaxScaler()
    df_cleaned[NUMERIC_FEATURES] = scaler.fit_transform(df_cleaned[NUMERIC_FEATURES])

    # Save fitted scaler
    scaler_path = artifacts_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted MinMaxScaler to: {scaler_path}")

    # 4. Class Balancing: SMOTE oversampling
    X = df_cleaned.drop(columns=[TARGET_COLUMN])
    y = df_cleaned[TARGET_COLUMN]

    smote = SMOTE(random_state=random_state)
    X_resampled_array, y_resampled_array = smote.fit_resample(X, y)

    X_resampled = pd.DataFrame(X_resampled_array, columns=X.columns)
    y_resampled = pd.Series(y_resampled_array, name=TARGET_COLUMN)

    # Round binary features after SMOTE interpolation back to clinical discrete flags (0 or 1)
    X_resampled[BINARY_FEATURES] = X_resampled[BINARY_FEATURES].round().astype(int)

    # Combine to represent the full SMOTE balanced cohort
    df_smote = pd.concat([X_resampled, y_resampled], axis=1)
    print(f"SMOTE cohort shape: {df_smote.shape}")
    print(f"SMOTE class distribution: {y_resampled.value_counts().to_dict()}")

    # 5. Data Splitting: 80/20 train-test split stratified on y
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.20,
        stratify=y_resampled,
        random_state=random_state,
    )

    print(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    print(f"Train labels shape: {y_train.shape}, Test labels shape: {y_test.shape}")

    # 6. Export Processed Datasets
    x_train_path = processed_dir / "X_train.csv"
    x_test_path = processed_dir / "X_test.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print(f"Saved processed datasets to: {processed_dir}")
    print(f"  - {x_train_path.name}")
    print(f"  - {x_test_path.name}")
    print(f"  - {y_train_path.name}")
    print(f"  - {y_test_path.name}")

    return X_train, X_test, y_train, y_test, scaler, df_raw, df_smote


def main():
    print("=" * 60)
    print("STEP 3: PREPROCESSING PIPELINE")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler, df_raw, df_smote = preprocess_pam_data()
    print("=" * 60)
    print("PIPELINE VERIFICATION SUMMARY:")
    print(f"  Raw Cohort Shape:   {df_raw.shape}")
    print(f"  SMOTE Cohort Shape: {df_smote.shape}")
    print(f"  X_train Shape:      {X_train.shape}")
    print(f"  X_test Shape:       {X_test.shape}")
    print(f"  y_train Shape:      {y_train.shape}")
    print(f"  y_test Shape:       {y_test.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
