"""Feature Optimization Engine for PAM Clinical Decision Support System.

This module implements the PAMFeatureSelector class executing a 3-layer optimization
strategy:
1. Layer 1: Chi-Square (χ²) test for categorical dependency.
2. Layer 2: Pearson correlation filtering to eliminate multicollinear numeric features.
3. Layer 3: Recursive Feature Elimination (RFE) to prune to the optimal 9 anchors.
"""

from typing import List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, RFE
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

DEFAULT_NUMERIC_FEATURES: List[str] = [
    "Age",
    "Symptom_Onset_Days",
    "CSF_Pressure",
    "CSF_WBC_Count",
]

DEFAULT_BINARY_FEATURES: List[str] = [
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


class PAMFeatureSelector:
    """Three-layer clinical feature selector for rare-disease diagnostic optimization.

    Executes statistical dependency filtering, collinearity pruning, and
    tree-based recursive feature elimination to reduce bedside survey fatigue
    while preserving maximal diagnostic signal.

    Attributes:
        random_state: Random seed for estimator reproducibility.
        selected_features_: List of top selected feature names after fitting.
        chi_scores_: DataFrame containing Layer 1 Chi-Square scores and p-values.
        dropped_collinear_: List of features dropped during Layer 2 correlation check.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initializes the PAMFeatureSelector.

        Args:
            random_state: Seed for pseudo-random number generator. Defaults to 42.
        """
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.chi_scores_: Optional[pd.DataFrame] = None
        self.dropped_collinear_: List[str] = []

    def chi_square_filter(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        categorical_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Calculates Chi-Square (χ²) scores and p-values for categorical features.

        Tests the null hypothesis of independence between categorical clinical signs
        and the binary outcome (PAM vs. Control).

        Args:
            X: Input feature DataFrame (values must be non-negative).
            y: Binary target vector (0 or 1).
            categorical_cols: Optional list of categorical column names.
                If None, uses default binary features present in X.

        Returns:
            pd.DataFrame: Table with columns ['Feature', 'Chi2_Score', 'p_value', 'Significant'].
        """
        cat_cols = categorical_cols or [c for c in DEFAULT_BINARY_FEATURES if c in X.columns]
        if not cat_cols:
            cat_cols = list(X.columns)

        X_cat = X[cat_cols]
        # Chi-Square requires non-negative inputs
        if (X_cat < 0).any().any():
            raise ValueError("Chi-Square input contains negative values. Features must be non-negative.")

        scores, p_values = chi2(X_cat, y)

        df_chi = pd.DataFrame({
            "Feature": cat_cols,
            "Chi2_Score": np.round(scores, 4),
            "p_value": p_values,
            "Significant (p < 0.05)": p_values < 0.05,
        }).sort_values(by="Chi2_Score", ascending=False).reset_index(drop=True)

        self.chi_scores_ = df_chi
        logger.info(
            f"Layer 1 (Chi-Square): Evaluated {len(cat_cols)} categorical features. "
            f"{df_chi['Significant (p < 0.05)'].sum()}/{len(cat_cols)} statistically significant."
        )
        return df_chi

    def correlation_filter(
        self,
        X: pd.DataFrame,
        threshold: float = 0.80,
        numerical_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Identifies and drops highly collinear continuous features (|r| > threshold).

        Prevents multicollinear inflation in downstream models and reduces redundant
        laboratory diagnostic order burden.

        Args:
            X: Input feature DataFrame.
            threshold: Absolute Pearson correlation threshold for redundancy. Defaults to 0.80.
            numerical_cols: Optional list of numeric column names to inspect.
                If None, uses default continuous features present in X.

        Returns:
            Tuple[pd.DataFrame, List[str]]:
                - Filtered DataFrame with redundant columns removed.
                - List of dropped feature names.
        """
        num_cols = numerical_cols or [c for c in DEFAULT_NUMERIC_FEATURES if c in X.columns]
        if len(num_cols) < 2:
            return X.copy(), []

        corr_matrix = X[num_cols].corr().abs()
        dropped_cols: List[str] = []

        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col_i, col_j = num_cols[i], num_cols[j]
                r_val = corr_matrix.loc[col_i, col_j]
                if r_val > threshold and col_j not in dropped_cols:
                    logger.info(f"Collinearity detected: |r({col_i}, {col_j})| = {r_val:.4f} > {threshold}. Dropping '{col_j}'.")
                    dropped_cols.append(col_j)

        self.dropped_collinear_ = dropped_cols
        X_filtered = X.drop(columns=dropped_cols)
        logger.info(f"Layer 2 (Correlation): Dropped {len(dropped_cols)} redundant feature(s): {dropped_cols}")
        return X_filtered, dropped_cols

    def rfe_filter(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        n_features: int = 9,
    ) -> List[str]:
        """Performs Recursive Feature Elimination (RFE) to isolate the top n_features.

        Uses an interpretable DecisionTreeClassifier estimator to prune features
        iteratively according to Gini importance ranking.

        Args:
            X: Input feature DataFrame.
            y: Binary target vector.
            n_features: Exact number of top features to retain. Defaults to 9.

        Returns:
            List[str]: Names of the selected optimal features.
        """
        if n_features > X.shape[1]:
            logger.warning(f"Requested {n_features} features, but input only has {X.shape[1]}. Retaining all.")
            self.selected_features_ = list(X.columns)
            return self.selected_features_

        estimator = DecisionTreeClassifier(random_state=self.random_state, max_depth=5)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)

        selected = X.columns[rfe.support_].tolist()
        self.selected_features_ = selected
        logger.info(f"Layer 3 (RFE): Isolated top {len(selected)} diagnostic anchors: {selected}")
        return selected

    def fit_select(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        n_features: int = 9,
        corr_threshold: float = 0.80,
    ) -> List[str]:
        """Chains all 3 optimization layers sequentially to extract the top features.

        Args:
            X: Input feature DataFrame.
            y: Binary target vector.
            n_features: Number of final features to retain via RFE. Defaults to 9.
            corr_threshold: Threshold for collinearity elimination. Defaults to 0.80.

        Returns:
            List[str]: Final list of selected diagnostic anchors.
        """
        # Layer 1: Chi-Square inspection (logged for transparency)
        self.chi_square_filter(X, y)

        # Layer 2: Multicollinearity filtering
        X_uncorr, _ = self.correlation_filter(X, threshold=corr_threshold)

        # Layer 3: RFE pruning
        top_features = self.rfe_filter(X_uncorr, y, n_features=n_features)
        return top_features
