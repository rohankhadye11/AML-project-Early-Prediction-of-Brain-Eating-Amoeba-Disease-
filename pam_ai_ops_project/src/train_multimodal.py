"""
Multimodal Fusion Model Training Script for PAM Clinical Decision Support System.
Module: src/train_multimodal.py

Fuses the 9 RFE-optimized clinical features with 50-dimensional neuroimaging
(MRI/CT) extracted features to train and evaluate a multimodal Random Forest Classifier.
Persists the trained model to models/artifacts/multimodal_model.pkl.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def simulate_mri_features(
    y: pd.Series,
    random_state: int = 42,
    shift_strength: float = 0.12,
    p_detect: float = 0.85,
) -> pd.DataFrame:
    """
    Generate 50 simulated continuous neuroimaging features representing
    the OpenCV MRI feature extraction pipeline (intensity histograms, quadrant stats, ROI moments).
    For PAM positive cases (y == 1), subtle upward shifts simulate the detection of
    cerebral edema and leptomeningeal inflammation.

    Parameters:
    -----------
    y : pd.Series
        Target labels (0 for Negative, 1 for PAM).
    random_state : int
        Seed for reproducibility.
    shift_strength : float
        Magnitude of lesion/edema signal shift.
    p_detect : float
        Imaging sensitivity probability for active edema lesions.

    Returns:
    --------
    pd.DataFrame
        50-dimensional simulated MRI feature DataFrame.
    """
    np.random.seed(random_state)
    n_samples = len(y)

    # Base distribution representing normalized MRI histogram and spatial statistics
    mri_matrix = np.random.normal(loc=0.15, scale=0.06, size=(n_samples, 50))

    # Targeted feature indices representing hyperintensity / edema markers
    # (e.g., upper histogram bins, central ROI mean, quadrant standard deviations)
    edema_indices = [10, 15, 20, 32, 33, 40, 48, 49]

    for i in range(n_samples):
        label = y.iloc[i]
        if label == 1 and np.random.rand() < p_detect:
            # Positive cases with visible neuroimaging findings of necrotizing encephalitis
            shift_noise = np.random.normal(shift_strength, 0.02, size=len(edema_indices))
            mri_matrix[i, edema_indices] += shift_noise

    mri_matrix = np.clip(mri_matrix, 0.0, 1.0)
    cols = [f"mri_feature_{idx:02d}" for idx in range(50)]
    return pd.DataFrame(mri_matrix, columns=cols)


def train_multimodal_pipeline(
    processed_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    random_state: int = 42,
):
    """
    Execute multimodal data fusion and train the multimodal Random Forest model.

    Parameters:
    -----------
    processed_dir : Path or str, optional
        Directory containing X_train_opt.csv, X_test_opt.csv, y_train.csv, y_test.csv.
    artifacts_dir : Path or str, optional
        Directory to persist multimodal_model.pkl.
    random_state : int
        Seed for reproducibility (default 42).

    Returns:
    --------
    tuple:
        (rf_multi, metrics_dict, X_train_multi, X_test_multi)
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

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Optimized Clinical Features (9 features)
    x_train_path = processed_dir / "X_train_opt.csv"
    x_test_path = processed_dir / "X_test_opt.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"

    X_train_opt = pd.read_csv(x_train_path)
    X_test_opt = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    print(f"Loaded clinical features: X_train_opt={X_train_opt.shape}, X_test_opt={X_test_opt.shape}")

    # 2. Simulate 50-D MRI Features
    print("Simulating 50-D neuroimaging feature extraction with clinical lesion dynamics...")
    mri_train = simulate_mri_features(y_train, random_state=random_state)
    mri_test = simulate_mri_features(y_test, random_state=random_state + 100)

    # 3. Data Fusion: Concatenate 9 Clinical + 50 MRI Features -> 59 Features
    X_train_multi = pd.concat([X_train_opt, mri_train], axis=1)
    X_test_multi = pd.concat([X_test_opt, mri_test], axis=1)

    print(f"Constructed Multimodal Feature Sets:")
    print(f"  X_train_multi shape: {X_train_multi.shape} (9 Clinical + 50 MRI = 59 Dimensions)")
    print(f"  X_test_multi shape:  {X_test_multi.shape} (9 Clinical + 50 MRI = 59 Dimensions)")

    # 4. Train Multimodal Random Forest Model
    print("\nTraining Multimodal Random Forest Classifier (n_estimators=100, random_state=42)...")
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_multi.fit(X_train_multi, y_train)

    # 5. Evaluate on Test Split
    y_pred = rf_multi.predict(X_test_multi)
    y_prob = rf_multi.predict_proba(X_test_multi)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
    }

    # 6. Save Artifact
    model_artifact_path = artifacts_dir / "multimodal_model.pkl"
    joblib.dump(rf_multi, model_artifact_path)
    print(f"\nSuccessfully saved trained multimodal model to: {model_artifact_path}")

    return rf_multi, metrics, X_train_multi, X_test_multi


def main():
    print("=" * 70)
    print("PHASE 4: MULTIMODAL FUSION TRAINING (CLINICAL + NEUROIMAGING)")
    print("=" * 70)
    rf_multi, metrics, X_tr_multi, X_te_multi = train_multimodal_pipeline()

    print("\n" + "=" * 70)
    print("MULTIMODAL MODEL EVALUATION METRICS (TEST SET):")
    print("=" * 70)
    print(f"  Accuracy:   {metrics['Accuracy'] * 100:6.2f}%")
    print(f"  Precision:  {metrics['Precision'] * 100:6.2f}%")
    print(f"  Recall:     {metrics['Recall'] * 100:6.2f}%")
    print(f"  F1-Score:   {metrics['F1-Score'] * 100:6.2f}%")
    print(f"  ROC-AUC:    {metrics['ROC-AUC']:6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
