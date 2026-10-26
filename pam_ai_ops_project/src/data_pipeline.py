"""Data Preprocessing Pipeline Module for PAM Clinical Decision Support System.

This module provides a research-grade, leakage-free clinical data preprocessing
pipeline for Primary Amoebic Meningoencephalitis (PAM). It performs imputation,
MinMax scaling, and SMOTE class balancing strictly on the training partition.
"""

from typing import List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Standard 13 clinical features schema
CONTINUOUS_FEATURES: List[str] = [
    "Age",
    "Symptom_Onset_Days",
    "CSF_Pressure",
    "CSF_WBC_Count",
]

BINARY_FEATURES: List[str] = [
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

ALL_CLINICAL_FEATURES: List[str] = BINARY_FEATURES + CONTINUOUS_FEATURES


def preprocess_clinical_data(
    df: pd.DataFrame,
    target_col: str = "Outcome",
    test_size: float = 0.20,
    random_state: int = 42,
    continuous_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocesses raw clinical data with strict leakage prevention and SMOTE balancing.

    This function implements a leakage-safe medical ML pipeline:
    1. Validates input schema and handles missing target columns.
    2. Performs stratified train-test splitting BEFORE any transformation.
    3. Imputes missing continuous features with the training median and binary
       features with the training mode.
    4. Fits MinMaxScaler on training continuous features and transforms both sets.
    5. Applies SMOTE oversampling exclusively to the training split.
    6. Re-discretizes interpolated binary features post-SMOTE to ensure valid flags.

    Args:
        df: Input pandas DataFrame containing clinical features and target label.
        target_col: Name of the binary target column (1 = PAM Positive, 0 = Control).
            Defaults to "Outcome".
        test_size: Proportion of dataset allocated to the test split. Defaults to 0.20.
        random_state: Seed for pseudo-random number generator for reproducibility.
            Defaults to 42.
        continuous_cols: Optional list of continuous feature names. If None, uses default.
        binary_cols: Optional list of binary feature names. If None, uses default.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - X_train: Preprocessed, SMOTE-balanced training feature DataFrame.
            - X_test: Preprocessed, un-oversampled test feature DataFrame.
            - y_train: SMOTE-balanced training target Series.
            - y_test: Un-oversampled test target Series.

    Raises:
        ValueError: If input DataFrame is empty or target column is missing.
        TypeError: If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, received {type(df).__name__}")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        # Fallback to check for 'Target' column if 'Outcome' is absent
        if "Target" in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Using 'Target' instead.")
            target_col = "Target"
        else:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    # Resolve feature groupings
    num_cols = continuous_cols or [c for c in CONTINUOUS_FEATURES if c in df.columns]
    bin_cols = binary_cols or [c for c in BINARY_FEATURES if c in df.columns]

    logger.info(f"Initiating preprocessing pipeline on cohort of shape {df.shape}")
    logger.info(f"Target column: '{target_col}' | Continuous features: {len(num_cols)} | Binary features: {len(bin_cols)}")

    # Separate features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Step 1: Stratified Train-Test Split (Zero Data Leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        f"Stratified split completed: Train={X_train_raw.shape[0]} samples, "
        f"Test={X_test_raw.shape[0]} samples"
    )

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # Step 2: Missing Value Imputation (Fit on Train ONLY)
    # Continuous features -> Median
    for col in num_cols:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # Binary features -> Mode
    for col in bin_cols:
        mode_series = X_train[col].mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else 0
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    # Step 3: Feature Scaling (MinMaxScaler fit on Train ONLY)
    if num_cols:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        logger.info(f"MinMax scaling applied to {len(num_cols)} continuous features.")

    # Step 4: Class Balancing via SMOTE (Applied to Train ONLY)
    pre_smote_counts = dict(y_train.value_counts())
    logger.info(f"Pre-SMOTE training class distribution: {pre_smote_counts}")

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    X_train_bal = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    y_train_bal = pd.Series(y_train_resampled, name=target_col)

    # Re-discretize binary features post-interpolation (0 or 1)
    if bin_cols:
        X_train_bal[bin_cols] = X_train_bal[bin_cols].round().astype(int)

    post_smote_counts = dict(y_train_bal.value_counts())
    logger.info(f"Post-SMOTE training class distribution: {post_smote_counts}")
    logger.info(
        f"Preprocessing complete: X_train shape={X_train_bal.shape}, "
        f"X_test shape={X_test.shape}"
    )

    return X_train_bal, X_test, y_train_bal, y_test
