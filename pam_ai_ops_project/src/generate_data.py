"""
Synthetic Data Generation Module for PAM Clinical Decision Support System.
Generates synthetic clinical records for Primary Amoebic Meningoencephalitis (PAM)
and negative control cohorts based on clinical epidemiology and CSF profiles.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_pam_synthetic_data(
    n_neg: int = 820,
    n_pos: int = 180,
    random_state: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic clinical records for PAM and negative cohorts.

    Parameters:
    -----------
    n_neg : int
        Number of negative control records (default 820).
    n_pos : int
        Number of positive PAM records (default 180).
    random_state : int
        Random seed for reproducibility (default 42).
    output_path : Path or str, optional
        Filepath to save the raw CSV.

    Returns:
    --------
    pd.DataFrame
        Cohort dataframe with shape (n_neg + n_pos, 14).
    """
    np.random.seed(random_state)

    # 1. Negative Cohort Generation (n_neg = 820)
    neg_freshwater = np.random.binomial(n=1, p=0.15, size=n_neg)
    neg_nasal = np.random.binomial(n=1, p=0.08, size=n_neg)
    neg_headache = np.random.binomial(n=1, p=0.45, size=n_neg)
    neg_fever = np.random.binomial(n=1, p=0.50, size=n_neg)
    neg_nausea = np.random.binomial(n=1, p=0.40, size=n_neg)
    neg_stiff_neck = np.random.binomial(n=1, p=0.20, size=n_neg)
    neg_confusion = np.random.binomial(n=1, p=0.20, size=n_neg)
    neg_seizures = np.random.binomial(n=1, p=0.08, size=n_neg)
    neg_photophobia = np.random.binomial(n=1, p=0.25, size=n_neg)

    neg_age = np.clip(
        np.round(np.random.normal(loc=42.0, scale=18.0, size=n_neg)), 1, 80
    ).astype(int)
    neg_onset = np.clip(
        np.round(np.random.normal(loc=5.5, scale=2.5, size=n_neg)), 1, 12
    ).astype(int)
    neg_csf_pressure = np.round(
        np.random.normal(loc=180.0, scale=35.0, size=n_neg), 1
    )
    neg_csf_wbc = np.round(
        np.clip(np.random.normal(loc=150.0, scale=45.0, size=n_neg), 10.0, None), 1
    )
    neg_target = np.zeros(n_neg, dtype=int)

    df_neg = pd.DataFrame(
        {
            "Freshwater_Exposure": neg_freshwater,
            "Nasal_Irrigation_History": neg_nasal,
            "Severe_Headache": neg_headache,
            "Fever": neg_fever,
            "Nausea_Vomiting": neg_nausea,
            "Stiff_Neck": neg_stiff_neck,
            "Confusion": neg_confusion,
            "Seizures": neg_seizures,
            "Photophobia": neg_photophobia,
            "Age": neg_age,
            "Symptom_Onset_Days": neg_onset,
            "CSF_Pressure": neg_csf_pressure,
            "CSF_WBC_Count": neg_csf_wbc,
            "Target": neg_target,
        }
    )

    # 2. Positive PAM Cohort Generation (n_pos = 180)
    # PAM positive cases: freshwater exposure heavily skewed to 1, elevated CSF vitals
    pos_freshwater = np.random.binomial(n=1, p=0.92, size=n_pos)
    pos_nasal = np.random.binomial(n=1, p=0.35, size=n_pos)
    pos_headache = np.random.binomial(n=1, p=0.90, size=n_pos)
    pos_fever = np.random.binomial(n=1, p=0.88, size=n_pos)
    pos_nausea = np.random.binomial(n=1, p=0.82, size=n_pos)
    pos_stiff_neck = np.random.binomial(n=1, p=0.90, size=n_pos)
    pos_confusion = np.random.binomial(n=1, p=0.78, size=n_pos)
    pos_seizures = np.random.binomial(n=1, p=0.35, size=n_pos)
    pos_photophobia = np.random.binomial(n=1, p=0.72, size=n_pos)

    pos_age = np.clip(
        np.round(np.random.normal(loc=23.0, scale=14.0, size=n_pos)), 1, 80
    ).astype(int)
    pos_onset = np.clip(
        np.round(np.random.normal(loc=3.8, scale=1.8, size=n_pos)), 1, 12
    ).astype(int)
    pos_csf_pressure = np.round(
        np.random.normal(loc=385.0, scale=45.0, size=n_pos), 1
    )
    pos_csf_wbc = np.round(
        np.clip(np.random.normal(loc=600.0, scale=100.0, size=n_pos), 100.0, None), 1
    )
    pos_target = np.ones(n_pos, dtype=int)

    df_pos = pd.DataFrame(
        {
            "Freshwater_Exposure": pos_freshwater,
            "Nasal_Irrigation_History": pos_nasal,
            "Severe_Headache": pos_headache,
            "Fever": pos_fever,
            "Nausea_Vomiting": pos_nausea,
            "Stiff_Neck": pos_stiff_neck,
            "Confusion": pos_confusion,
            "Seizures": pos_seizures,
            "Photophobia": pos_photophobia,
            "Age": pos_age,
            "Symptom_Onset_Days": pos_onset,
            "CSF_Pressure": pos_csf_pressure,
            "CSF_WBC_Count": pos_csf_wbc,
            "Target": pos_target,
        }
    )

    # 3. Model Clinical Diagnostic Overlap (Atypical Presentations)
    # Simulates the reality where a small fraction of severe meningitis mimics
    # overlap with atypical/early PAM cases, maintaining exact 820:180 ratio.
    df_combined = pd.concat([df_neg, df_pos], ignore_index=True)
    idx_p = np.random.choice(df_combined[df_combined["Target"] == 1].index, size=39, replace=False)
    idx_n = np.random.choice(df_combined[df_combined["Target"] == 0].index, size=39, replace=False)
    df_combined.loc[idx_p, "Target"] = 0
    df_combined.loc[idx_n, "Target"] = 1

    df_cohort = (
        df_combined
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    # 4. Save to Disk if output path is provided
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_cohort.to_csv(out_p, index=False)
        print(f"Dataset successfully saved to: {out_p}")

    return df_cohort


def main():
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    raw_data_path = project_root / "data" / "raw" / "synthetic_pam_raw_1000.csv"

    print("=" * 60)
    print("STEP 2: SYNTHETIC DATA GENERATION")
    print("=" * 60)

    df = generate_pam_synthetic_data(
        n_neg=820,
        n_pos=180,
        random_state=42,
        output_path=raw_data_path,
    )

    print(f"Generated raw cohort shape: {df.shape}")
    print(f"Target distribution:\n{df['Target'].value_counts().to_dict()}")
    print("Feature columns (14 total):")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx:02d}. {col} ({df[col].dtype})")
    print("=" * 60)


if __name__ == "__main__":
    main()
