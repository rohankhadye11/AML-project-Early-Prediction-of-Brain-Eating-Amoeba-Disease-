"""
Feature Optimization Pipeline for PAM Clinical Decision Support System.
Applies a three-layer feature optimization strategy:
1. Chi-Square Test: Identifies statistically significant features (p < 0.05).
2. Correlation Analysis: Computes Pearson correlation to drop collinear numeric features (|r| > 0.8).
3. Recursive Feature Elimination (RFE): Prunes features down to 9 optimal predictors using DecisionTreeClassifier.
Exports:
- data/processed/X_train_opt.csv
- data/processed/X_test_opt.csv
- models/artifacts/selected_features.pkl
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, RFE
from sklearn.tree import DecisionTreeClassifier

NUMERIC_FEATURES = ["Age", "Symptom_Onset_Days", "CSF_Pressure", "CSF_WBC_Count"]


def run_feature_optimization(
    processed_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    n_features_to_select: int = 9,
    corr_threshold: float = 0.8,
    random_state: int = 42,
):
    """
    Execute three-layer feature optimization strategy.

    Parameters:
    -----------
    processed_dir : Path or str, optional
        Path to directory containing processed train/test datasets.
    artifacts_dir : Path or str, optional
        Path to directory for saving model artifacts.
    n_features_to_select : int
        Number of final features to retain via RFE (default 9).
    corr_threshold : float
        Absolute Pearson correlation threshold for dropping collinear features (default 0.8).
    random_state : int
        Random seed for reproducibility (default 42).

    Returns:
    --------
    tuple:
        (X_train_opt, X_test_opt, selected_features, df_chi, corr_matrix, drop_cols)
    """
    # 0. Resolve Paths
    project_root = Path(__file__).resolve().parent.parent
    if processed_dir is None:
        processed_dir = project_root / "data" / "processed"
    else:
        processed_dir = Path(processed_dir)

    if artifacts_dir is None:
        artifacts_dir = project_root / "models" / "artifacts"
    else:
        artifacts_dir = Path(artifacts_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Processed Datasets
    x_train_path = processed_dir / "X_train.csv"
    x_test_path = processed_dir / "X_test.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    print(f"Loaded training data: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Loaded testing data:  X_test={X_test.shape}, y_test={y_test.shape}")

    # =========================================================================
    # LAYER 1: Chi-Square Test for Independence
    # =========================================================================
    print("\n--- Layer 1: Chi-Square Test for Statistical Significance ---")
    chi_scores, p_values = chi2(X_train, y_train)
    df_chi = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Chi2_Score": chi_scores,
            "p_value": p_values,
            "Significant_p_lt_0.05": p_values < 0.05,
        }
    ).sort_values("Chi2_Score", ascending=False).reset_index(drop=True)

    significant_features = df_chi[df_chi["Significant_p_lt_0.05"]]["Feature"].tolist()
    print(f"Features with p < 0.05 ({len(significant_features)}/{len(X_train.columns)}):")
    for _, row in df_chi.iterrows():
        sig_flag = "SIGNIFICANT (p < 0.05)" if row["Significant_p_lt_0.05"] else "NOT SIGNIFICANT"
        print(f"  {row['Feature']:26}: Chi2={row['Chi2_Score']:8.2f}, p={row['p_value']:.4e} -> {sig_flag}")

    # =========================================================================
    # LAYER 2: Correlation Analysis (Multicollinearity Removal)
    # =========================================================================
    print("\n--- Layer 2: Correlation Analysis (Numeric Collinearity) ---")
    num_cols = [col for col in NUMERIC_FEATURES if col in X_train.columns]
    corr_matrix = X_train[num_cols].corr()
    print("Pearson Correlation Matrix (Numeric Biomarkers):")
    print(corr_matrix.round(4).to_string())

    drop_cols = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            r_val = corr_matrix.loc[c1, c2]
            if abs(r_val) > corr_threshold:
                print(f"  Collinearity detected: |r({c1}, {c2})| = {abs(r_val):.4f} > {corr_threshold}")
                if c2 not in drop_cols:
                    drop_cols.append(c2)

    if drop_cols:
        print(f"  Dropping highly correlated numeric feature(s): {drop_cols}")
        X_train_filtered = X_train.drop(columns=drop_cols)
        X_test_filtered = X_test.drop(columns=drop_cols)
    else:
        print("  No numeric pairs exceeded correlation threshold.")
        X_train_filtered = X_train.copy()
        X_test_filtered = X_test.copy()

    # =========================================================================
    # LAYER 3: Recursive Feature Elimination (RFE)
    # =========================================================================
    print(f"\n--- Layer 3: Recursive Feature Elimination (RFE -> {n_features_to_select} Features) ---")
    base_estimator = DecisionTreeClassifier(random_state=random_state)
    rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X_train_filtered, y_train)

    selected_mask = rfe.support_
    selected_features = X_train_filtered.columns[selected_mask].tolist()

    print(f"Selected {len(selected_features)} optimal features by RFE:")
    for idx, feat in enumerate(selected_features, 1):
        print(f"  {idx:02d}. {feat}")

    # Transform datasets
    X_train_opt = X_train_filtered[selected_features].copy()
    X_test_opt = X_test_filtered[selected_features].copy()

    # =========================================================================
    # EXPORT & ARTIFACT PERSISTENCE
    # =========================================================================
    train_opt_path = processed_dir / "X_train_opt.csv"
    test_opt_path = processed_dir / "X_test_opt.csv"
    X_train_opt.to_csv(train_opt_path, index=False)
    X_test_opt.to_csv(test_opt_path, index=False)
    print(f"\nSaved optimized datasets:")
    print(f"  - {train_opt_path} (Shape: {X_train_opt.shape})")
    print(f"  - {test_opt_path}  (Shape: {X_test_opt.shape})")

    features_pkl_path = artifacts_dir / "selected_features.pkl"
    joblib.dump(selected_features, features_pkl_path)
    print(f"Saved selected feature list to: {features_pkl_path}")

    return X_train_opt, X_test_opt, selected_features, df_chi, corr_matrix, drop_cols


def main():
    print("=" * 60)
    print("PHASE 2 - STEP 1: FEATURE OPTIMIZATION PIPELINE")
    print("=" * 60)
    run_feature_optimization()
    print("=" * 60)


if __name__ == "__main__":
    main()
