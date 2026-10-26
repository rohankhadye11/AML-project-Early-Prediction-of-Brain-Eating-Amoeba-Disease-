"""Ablation Study Engine for PAM Clinical Decision Support System.

Executes a 2x2 factorial ablation experiment benchmarking a constant Random Forest
classifier under four distinct data-engineering and feature-selection conditions:
- Test A (Baseline):          Raw Imbalanced Data + All 13 Features
- Test B (Data Ablation):     SMOTE Balanced Data + All 13 Features
- Test C (Feature Ablation):  Raw Imbalanced Data + Top 9 RFE Features
- Test D (Proposed Optimal):  SMOTE Balanced Data + Top 9 RFE Features

Evaluates Accuracy, Precision, Recall (Sensitivity), F1-Score, ROC-AUC, and False Negatives.
"""

from typing import Dict, List, Union
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def run_ablation_study(
    X_train_raw: pd.DataFrame,
    y_train_raw: Union[pd.Series, np.ndarray],
    X_train_smote: pd.DataFrame,
    y_train_smote: Union[pd.Series, np.ndarray],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    top_9_feature_names: List[str],
    random_state: int = 42,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """Trains and benchmarks Random Forest across 4 controlled ablation conditions.

    Ablation Design:
    ================
    Condition                       | Balancing Strategy | Feature Dimensionality
    --------------------------------+--------------------+-----------------------
    Test A (Baseline)               | Raw Imbalanced     | All 13 Features
    Test B (Data Ablation)          | SMOTE Balanced     | All 13 Features
    Test C (Feature Ablation)       | Raw Imbalanced     | Top 9 RFE Features
    Test D (Proposed Optimal)       | SMOTE Balanced     | Top 9 RFE Features

    Scientific Rationale:
    - Comparing Test A vs. Test B isolates the independent benefit of SMOTE oversampling.
    - Comparing Test A vs. Test C isolates the independent benefit of RFE feature pruning.
    - Comparing Test B vs. Test D proves whether feature selection still improves balanced models.
    - Comparing Test A vs. Test D demonstrates the cumulative synergistic performance gain.

    Args:
        X_train_raw: Imbalanced, scaled training features (All 13 columns).
        y_train_raw: Imbalanced training target vector.
        X_train_smote: SMOTE-balanced, scaled training features (All 13 columns).
        y_train_smote: SMOTE-balanced training target vector.
        X_test: Clean test feature DataFrame (All 13 columns).
        y_test: Clean test target vector.
        top_9_feature_names: Exact list of the 9 RFE-selected feature names.
        random_state: Random seed for RandomForestClassifier. Defaults to 42.
        n_estimators: Number of trees in Random Forest ensemble. Defaults to 100.

    Returns:
        pd.DataFrame: Comparative benchmark metrics table across all 4 experimental conditions.

    Raises:
        ValueError: If top_9_feature_names columns are not present in X_train.
    """
    logger.info("=" * 80)
    logger.info("STARTING PHASE 2 ABLATION STUDY BENCHMARK")
    logger.info("=" * 80)

    # Validate feature names
    missing = set(top_9_feature_names) - set(X_train_raw.columns)
    if missing:
        raise ValueError(f"Features {missing} not found in training dataset columns: {list(X_train_raw.columns)}")

    # 1. Define the 4 experimental configurations
    ablation_experiments = {
        "Test A (Baseline)": {
            "description": "Raw Imbalanced + All 13 Features",
            "X_train": X_train_raw,
            "y_train": y_train_raw,
            "X_test": X_test,
            "feature_count": X_train_raw.shape[1],
            "balancing": "None (Raw Imbalanced)",
        },
        "Test B (Data Ablation)": {
            "description": "SMOTE Balanced + All 13 Features",
            "X_train": X_train_smote,
            "y_train": y_train_smote,
            "X_test": X_test,
            "feature_count": X_train_smote.shape[1],
            "balancing": "SMOTE Oversampling",
        },
        "Test C (Feature Ablation)": {
            "description": "Raw Imbalanced + Top 9 RFE Features",
            "X_train": X_train_raw[top_9_feature_names],
            "y_train": y_train_raw,
            "X_test": X_test[top_9_feature_names],
            "feature_count": len(top_9_feature_names),
            "balancing": "None (Raw Imbalanced)",
        },
        "Test D (Proposed Optimal)": {
            "description": "SMOTE Balanced + Top 9 RFE Features",
            "X_train": X_train_smote[top_9_feature_names],
            "y_train": y_train_smote,
            "X_test": X_test[top_9_feature_names],
            "feature_count": len(top_9_feature_names),
            "balancing": "SMOTE Oversampling",
        },
    }

    results: List[Dict[str, Union[str, float, int]]] = []

    # 2. Execute training and evaluation loop under identical hyperparameters
    for condition_name, config in ablation_experiments.items():
        logger.info(f"Evaluating {condition_name}: {config['description']}")

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1,
        )

        clf.fit(config["X_train"], config["y_train"])

        y_pred = clf.predict(config["X_test"])
        y_prob = clf.predict_proba(config["X_test"])[:, 1]

        # Calculate clinical metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Experiment Condition": condition_name,
            "Balancing Strategy": config["balancing"],
            "Features": f"{config['feature_count']} features",
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall (Sensitivity)": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "False Negatives (Fatal Misses)": int(fn),
            "True Positives (Detected PAM)": int(tp),
        })

    df_results = pd.DataFrame(results)
    logger.info("=" * 80)
    logger.info("ABLATION STUDY COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    return df_results
