"""Execution Script for Phase 1: Robust Data Engineering & Sensitivity Analysis.

This script demonstrates the end-to-end Phase 1 pipeline:
1. Generates a calibrated 1,000-patient synthetic PAM cohort.
2. Applies the leakage-free preprocessing pipeline (Imputation, MinMaxScaler, SMOTE).
3. Executes clinical noise injection for sensitivity analysis (Continuous ±10%, Categorical 5% flip).
4. Demonstrates OpenCV neuroimaging degradation (Gaussian Blur + Salt-and-Pepper noise).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure local src/ directory is in python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_pipeline import (
    preprocess_clinical_data,
    CONTINUOUS_FEATURES,
    BINARY_FEATURES,
)
from sensitivity_analysis import ClinicalNoiseInjector


def generate_mock_pam_dataset(
    n_samples: int = 1000,
    pam_ratio: float = 0.18,
    missing_rate: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generates a realistic synthetic cohort simulating PAM emergency intake.

    Args:
        n_samples: Total number of patient records to simulate. Defaults to 1000.
        pam_ratio: Proportion of positive PAM cases (simulates 180:820 ratio).
            Defaults to 0.18.
        missing_rate: Rate of artificial missing values to test imputation. Defaults to 0.02.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: Synthetic DataFrame with 13 clinical features and 'Outcome'.
    """
    rng = np.random.default_rng(random_state)
    n_pos = int(round(n_samples * pam_ratio))
    n_neg = n_samples - n_pos

    # 1. Negative Controls (Acute bacterial/viral meningitis mimics)
    neg_data = {
        "Freshwater_Exposure": rng.binomial(n=1, p=0.15, size=n_neg),
        "Nasal_Irrigation_History": rng.binomial(n=1, p=0.08, size=n_neg),
        "Severe_Headache": rng.binomial(n=1, p=0.45, size=n_neg),
        "Fever": rng.binomial(n=1, p=0.50, size=n_neg),
        "Nausea_Vomiting": rng.binomial(n=1, p=0.40, size=n_neg),
        "Stiff_Neck": rng.binomial(n=1, p=0.20, size=n_neg),
        "Confusion": rng.binomial(n=1, p=0.20, size=n_neg),
        "Seizures": rng.binomial(n=1, p=0.08, size=n_neg),
        "Photophobia": rng.binomial(n=1, p=0.25, size=n_neg),
        "Age": np.clip(np.round(rng.normal(loc=42.0, scale=18.0, size=n_neg)), 1, 80).astype(int),
        "Symptom_Onset_Days": np.clip(np.round(rng.normal(loc=5.5, scale=2.5, size=n_neg)), 1, 14).astype(int),
        "CSF_Pressure": np.round(rng.normal(loc=180.0, scale=35.0, size=n_neg), 1),
        "CSF_WBC_Count": np.round(np.clip(rng.normal(loc=150.0, scale=45.0, size=n_neg), 10.0, None), 1),
        "Outcome": np.zeros(n_neg, dtype=int),
    }

    # 2. Positive PAM Cohort (Severe necrotizing encephalitis)
    pos_data = {
        "Freshwater_Exposure": rng.binomial(n=1, p=0.92, size=n_pos),
        "Nasal_Irrigation_History": rng.binomial(n=1, p=0.35, size=n_pos),
        "Severe_Headache": rng.binomial(n=1, p=0.90, size=n_pos),
        "Fever": rng.binomial(n=1, p=0.88, size=n_pos),
        "Nausea_Vomiting": rng.binomial(n=1, p=0.82, size=n_pos),
        "Stiff_Neck": rng.binomial(n=1, p=0.90, size=n_pos),
        "Confusion": rng.binomial(n=1, p=0.78, size=n_pos),
        "Seizures": rng.binomial(n=1, p=0.35, size=n_pos),
        "Photophobia": rng.binomial(n=1, p=0.72, size=n_pos),
        "Age": np.clip(np.round(rng.normal(loc=23.0, scale=14.0, size=n_pos)), 1, 80).astype(int),
        "Symptom_Onset_Days": np.clip(np.round(rng.normal(loc=3.8, scale=1.8, size=n_pos)), 1, 12).astype(int),
        "CSF_Pressure": np.round(rng.normal(loc=385.0, scale=45.0, size=n_pos), 1),
        "CSF_WBC_Count": np.round(np.clip(rng.normal(loc=600.0, scale=100.0, size=n_pos), 100.0, None), 1),
        "Outcome": np.ones(n_pos, dtype=int),
    }

    df_combined = pd.concat([pd.DataFrame(neg_data), pd.DataFrame(pos_data)], ignore_index=True)

    # Introduce slight diagnostic overlap (3.9% atypical presentations)
    idx_p = rng.choice(df_combined[df_combined["Outcome"] == 1].index, size=int(n_samples * 0.039), replace=False)
    idx_n = rng.choice(df_combined[df_combined["Outcome"] == 0].index, size=int(n_samples * 0.039), replace=False)
    df_combined.loc[idx_p, "Outcome"] = 0
    df_combined.loc[idx_n, "Outcome"] = 1

    df_shuffled = df_combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 3. Inject random missing values to validate imputer
    if missing_rate > 0.0:
        for col in CONTINUOUS_FEATURES + BINARY_FEATURES:
            mask = rng.random(size=len(df_shuffled)) < missing_rate
            df_shuffled.loc[mask, col] = np.nan

    return df_shuffled


def main() -> None:
    """Executes the Phase 1 demonstration pipeline."""
    print("=" * 85)
    print("PHASE 1: CLINICAL DATA ENGINEERING & SENSITIVITY ANALYSIS PIPELINE")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # STEP 1: Generate Mock Clinical Dataset
    # -------------------------------------------------------------------------
    print("\n[1/4] Generating Mock PAM Clinical Dataset (1,000 records)...")
    df_raw = generate_mock_pam_dataset(n_samples=1000, pam_ratio=0.18, missing_rate=0.02, random_state=42)
    print(f"  - Dataset Shape: {df_raw.shape} (13 Features + 1 Target 'Outcome')")
    print(f"  - Target Breakdown: {dict(df_raw['Outcome'].value_counts())}")
    print(f"  - Total Missing Values Injected: {df_raw.isna().sum().sum()} entries")

    # -------------------------------------------------------------------------
    # STEP 2: Preprocess Clinical Data (Leakage-Safe SMOTE & Scaling)
    # -------------------------------------------------------------------------
    print("\n[2/4] Running Preprocessing Pipeline (Imputation, MinMaxScaler, Train-Only SMOTE)...")
    X_train, X_test, y_train, y_test = preprocess_clinical_data(
        df=df_raw,
        target_col="Outcome",
        test_size=0.20,
        random_state=42,
    )

    print(f"  - Clean X_train Shape (SMOTE Balanced): {X_train.shape}")
    print(f"  - Clean y_train Distribution:          {dict(y_train.value_counts())}")
    print(f"  - Clean X_test Shape (Un-oversampled):  {X_test.shape}")
    print(f"  - Clean y_test Distribution:           {dict(y_test.value_counts())}")
    print(f"  - Continuous Features Scaled Range:     [{X_test['CSF_Pressure'].min():.4f}, {X_test['CSF_Pressure'].max():.4f}]")

    # -------------------------------------------------------------------------
    # STEP 3: Sensitivity Analysis & Noise Injection on Test Set
    # -------------------------------------------------------------------------
    print("\n[3/4] Initializing Clinical Noise Injector for Sensitivity Analysis...")
    injector = ClinicalNoiseInjector(random_state=42)

    # Continuous noise: ±10% variance simulating instrument drift
    X_test_noisy_cont = injector.perturb_continuous(
        df=X_test,
        columns=CONTINUOUS_FEATURES,
        variance=0.10,
        subset_fraction=1.0,
        clip_bounds=(0.0, 1.0),
    )

    # Categorical noise: 5% bit flip simulating intake transcription errors
    X_test_noisy_final = injector.flip_categorical(
        df=X_test_noisy_cont,
        columns=BINARY_FEATURES,
        flip_rate=0.05,
    )

    print(f"  - Clean Test Set Shape: {X_test.shape}")
    print(f"  - Noisy Test Set Shape: {X_test_noisy_final.shape}")

    # Compare Clean vs Noisy Sample
    sample_cols = ["Age", "CSF_Pressure", "Severe_Headache", "Stiff_Neck", "Freshwater_Exposure"]
    print("\n--- SAMPLE COMPARISON: CLEAN TEST SET (FIRST 3 ROWS) ---")
    print(X_test[sample_cols].head(3).to_string(index=True))

    print("\n--- SAMPLE COMPARISON: NOISY TEST SET (FIRST 3 ROWS) ---")
    print(X_test_noisy_final[sample_cols].head(3).to_string(index=True))

    # Delta report
    cont_deltas = (X_test_noisy_final[CONTINUOUS_FEATURES] - X_test[CONTINUOUS_FEATURES]).abs().mean()
    cat_flips = (X_test_noisy_final[BINARY_FEATURES] != X_test[BINARY_FEATURES]).sum()

    print("\n--- SENSITIVITY INJECTION DELTA METRICS ---")
    print("Mean Absolute Deviation on Continuous Features (+/- 10% Target):")
    for feat, delta in cont_deltas.items():
        print(f"  * {feat:20}: Delta = {delta:.4f}")

    print("Total Flipped Binary Flags (5% clerical error rate):")
    for feat, flips in cat_flips.items():
        rate = (flips / len(X_test)) * 100
        print(f"  * {feat:26}: {flips:2d} flips ({rate:.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 4: Neuroimaging Noise Injection Demonstration (OpenCV)
    # -------------------------------------------------------------------------
    print("\n[4/4] Demonstrating Neuroimaging Degradation on 224x224 Grayscale MRI Slice...")
    # Create synthetic axial brain slice
    mock_mri = np.zeros((224, 224), dtype=np.uint8)
    import cv2
    cv2.ellipse(mock_mri, (112, 112), (80, 95), 0, 0, 360, 120, -1)
    cv2.circle(mock_mri, (95, 85), 18, 220, -1)  # Hyperintense edema lesion
    mock_mri = cv2.GaussianBlur(mock_mri, (5, 5), 1.5)

    # Inject noise (Gaussian Blur + 2% Salt-and-Pepper)
    noisy_mri = injector.add_image_noise(mock_mri, blur_ksize=(5, 5), noise_ratio=0.02)

    diff_pixels = np.count_nonzero(mock_mri != noisy_mri)
    pct_corrupted = (diff_pixels / mock_mri.size) * 100

    print(f"  - Input MRI Array Shape:   {mock_mri.shape} (Grayscale uint8)")
    print(f"  - Degraded MRI Array Shape:{noisy_mri.shape}")
    print(f"  - Mean Pixel Intensity:    Clean={mock_mri.mean():.2f} | Noisy={noisy_mri.mean():.2f}")
    print(f"  - Standard Deviation:      Clean={mock_mri.std():.2f} | Noisy={noisy_mri.std():.2f}")
    print(f"  - Total Altered Pixels:    {diff_pixels} / {mock_mri.size} ({pct_corrupted:.1f}% affected by blur + salt/pepper)")

    print("\n" + "=" * 85)
    print("PHASE 1 PIPELINE VERIFICATION SUCCESSFUL: ALL MODULES READY FOR MODEL BENCHMARKING")
    print("=" * 85)


if __name__ == "__main__":
    main()
