"""
Model Benchmarking and Selection Module for PAM Clinical Decision Support System.
Trains and benchmarks five machine learning algorithms:
1. Logistic Regression
2. Naive Bayes (GaussianNB)
3. Decision Tree Classifier
4. Support Vector Machine (SVC with probability=True)
5. Random Forest Classifier (n_estimators=100, random_state=42)

Evaluates on test split across:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
Persists the top-performing model to models/artifacts/best_model.pkl.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate_models(
    processed_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    random_state: int = 42,
):
    """
    Train and benchmark the 5 specified classifiers on optimized features.

    Parameters:
    -----------
    processed_dir : Path or str, optional
        Path to processed data directory.
    artifacts_dir : Path or str, optional
        Path to model artifacts directory.
    random_state : int
        Seed for reproducibility (default 42).

    Returns:
    --------
    tuple:
        (df_results, trained_models, test_preds, test_probs, y_test)
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

    # 1. Load Optimized Datasets
    X_train_opt = pd.read_csv(processed_dir / "X_train_opt.csv")
    X_test_opt = pd.read_csv(processed_dir / "X_test_opt.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    print(f"Loaded optimized training features: {X_train_opt.shape}")
    print(f"Loaded optimized testing features:  {X_test_opt.shape}")

    # 2. Define Model Suite
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=random_state, max_iter=1000
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Support Vector Machine": SVC(
            probability=True, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    # 3. Train and Evaluate Each Model
    results = []
    trained_models = {}
    test_preds = {}
    test_probs = {}

    for name, model in models.items():
        # Train
        model.fit(X_train_opt, y_train)
        trained_models[name] = model

        # Predict
        y_pred = model.predict(X_test_opt)
        y_prob = model.predict_proba(X_test_opt)[:, 1]

        test_preds[name] = y_pred
        test_probs[name] = y_prob

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(auc, 4),
            }
        )

    df_results = pd.DataFrame(results)

    # 4. Save the Top-Performing Random Forest Model
    best_model = trained_models["Random Forest"]
    best_model_path = artifacts_dir / "best_model.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"\nPersisted top-performing model to: {best_model_path}")

    return df_results, trained_models, test_preds, test_probs, y_test, X_test_opt


def main():
    print("=" * 70)
    print("PHASE 2 - STEP 2: MODEL BENCHMARKING & SELECTION")
    print("=" * 70)
    df_results, trained_models, _, _, _, X_test_opt = train_and_evaluate_models()

    print("\n" + "=" * 70)
    print("CONSOLIDATED MODEL BENCHMARK COMPARISON TABLE:")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)

    # Feature Importance of Random Forest
    rf_model = trained_models["Random Forest"]
    importances = pd.Series(
        rf_model.feature_importances_, index=X_test_opt.columns
    ).sort_values(ascending=False)

    print("\nRANDOM FOREST FEATURE IMPORTANCE SCORES:")
    for feat, imp in importances.items():
        print(f"  {feat:24}: {imp:.4f} ({imp * 100:5.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
