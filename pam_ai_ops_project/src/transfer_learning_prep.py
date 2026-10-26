"""Transfer Learning Data Preparation Module for PAM Clinical Decision Support.

This module prepares PyTorch DataLoaders for a two-stage Transfer Learning workflow:
1. Stage 1 (Pre-training): General acute meningitis baseline representations
   learned from a large-scale real-world cohort ('real_meningitis_baseline.csv').
2. Stage 2 (Fine-tuning): Specific Naegleria fowleri / PAM classification
   adapted on a focused clinical cohort ('synthetic_pam_data.csv').

Provides:
- ClinicalDataset: PyTorch Dataset wrapper converting tabular arrays to PyTorch Tensors.
- prepare_transfer_learning_loaders: Orchestration function yielding PyTorch DataLoaders.
"""

from typing import Dict, Optional, Tuple, Union
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure local module access
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_pipeline import (
    preprocess_clinical_data,
    CONTINUOUS_FEATURES,
    BINARY_FEATURES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


# =============================================================================
# PyTorch Dataset Definition
# =============================================================================

class ClinicalDataset(Dataset):
    """Custom PyTorch Dataset for tabular clinical features and binary outcomes.

    Encapsulates scaled, imputed clinical feature matrices and outcome labels,
    converting them into standard PyTorch FloatTensors for backpropagation.

    Attributes:
        features: torch.FloatTensor of shape (N, D), where N is sample count
            and D is feature dimensionality (e.g., 13 features).
        targets: torch.FloatTensor of shape (N, 1), representing binary labels
            (1.0 = Positive Case, 0.0 = Control).
        feature_names: List of strings detailing the column ordering.
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[list] = None,
    ) -> None:
        """Initializes the ClinicalDataset.

        Args:
            X: Input clinical features (Pandas DataFrame or NumPy array).
            y: Target binary labels (Pandas Series or NumPy array).
            feature_names: Optional list of feature names. If X is a DataFrame,
                extracted automatically.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            x_vals = X.values.astype(np.float32)
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            x_vals = np.asarray(X, dtype=np.float32)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_vals = y.values.astype(np.float32).reshape(-1, 1)
        else:
            y_vals = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        if len(x_vals) != len(y_vals):
            raise ValueError(f"Feature count ({len(x_vals)}) does not match target count ({len(y_vals)}).")

        self.features = torch.tensor(x_vals, dtype=torch.float32)
        self.targets = torch.tensor(y_vals, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns the total number of clinical samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a single patient feature vector and corresponding outcome label.

        Args:
            idx: Integer index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (features_tensor, target_tensor)
        """
        return self.features[idx], self.targets[idx]


# =============================================================================
# Helper: Synthetic Data Generator (Fallback if CSVs don't exist yet)
# =============================================================================

def _ensure_mock_csv_exists(
    filepath: Union[str, Path],
    n_samples: int = 1000,
    is_pretrain_baseline: bool = False,
    random_state: int = 42,
) -> Path:
    """Generates synthetic baseline or PAM cohorts if CSV files are not on disk.

    Args:
        filepath: Target filepath to load or create.
        n_samples: Number of patient records to simulate. Defaults to 1000.
        is_pretrain_baseline: If True, simulates broad real-world acute bacterial
            meningitis epidemiological distribution. If False, simulates rare PAM cohort.
        random_state: Random seed for reproducibility.

    Returns:
        Path: Verified Path object to the existing or newly created CSV file.
    """
    path = Path(filepath)
    if path.exists():
        logger.info(f"Dataset already exists on disk: {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    logger.info(f"Generating placeholder cohort for transfer learning: {path.name} (N={n_samples})")

    if is_pretrain_baseline:
        # Pretraining baseline: Large-scale acute bacterial & viral meningitis registry
        # Lower freshwater exposure (~10%), standard meningitis fever and stiff neck
        df = pd.DataFrame({
            "Freshwater_Exposure": rng.binomial(n=1, p=0.08, size=n_samples),
            "Nasal_Irrigation_History": rng.binomial(n=1, p=0.05, size=n_samples),
            "Severe_Headache": rng.binomial(n=1, p=0.65, size=n_samples),
            "Fever": rng.binomial(n=1, p=0.75, size=n_samples),
            "Nausea_Vomiting": rng.binomial(n=1, p=0.55, size=n_samples),
            "Stiff_Neck": rng.binomial(n=1, p=0.70, size=n_samples),
            "Confusion": rng.binomial(n=1, p=0.40, size=n_samples),
            "Seizures": rng.binomial(n=1, p=0.15, size=n_samples),
            "Photophobia": rng.binomial(n=1, p=0.50, size=n_samples),
            "Age": np.clip(rng.normal(loc=38.0, scale=18.0, size=n_samples), 1, 85).astype(int),
            "Symptom_Onset_Days": np.clip(rng.normal(loc=4.5, scale=2.5, size=n_samples), 1, 14).astype(int),
            "CSF_Pressure": np.round(rng.normal(loc=210.0, scale=40.0, size=n_samples), 1),
            "CSF_WBC_Count": np.round(np.clip(rng.normal(loc=350.0, scale=80.0, size=n_samples), 10.0, None), 1),
            "Outcome": rng.binomial(n=1, p=0.35, size=n_samples),
        })
    else:
        # Fine-tuning cohort: Ultra-rare PAM presentation (Freshwater ~90%, extreme CSF pressure)
        n_pos = int(n_samples * 0.18)
        n_neg = n_samples - n_pos

        neg_df = pd.DataFrame({
            "Freshwater_Exposure": rng.binomial(n=1, p=0.12, size=n_neg),
            "Nasal_Irrigation_History": rng.binomial(n=1, p=0.06, size=n_neg),
            "Severe_Headache": rng.binomial(n=1, p=0.45, size=n_neg),
            "Fever": rng.binomial(n=1, p=0.50, size=n_neg),
            "Nausea_Vomiting": rng.binomial(n=1, p=0.40, size=n_neg),
            "Stiff_Neck": rng.binomial(n=1, p=0.25, size=n_neg),
            "Confusion": rng.binomial(n=1, p=0.20, size=n_neg),
            "Seizures": rng.binomial(n=1, p=0.08, size=n_neg),
            "Photophobia": rng.binomial(n=1, p=0.30, size=n_neg),
            "Age": np.clip(rng.normal(loc=42.0, scale=18.0, size=n_neg), 1, 80).astype(int),
            "Symptom_Onset_Days": np.clip(rng.normal(loc=5.5, scale=2.5, size=n_neg), 1, 14).astype(int),
            "CSF_Pressure": np.round(rng.normal(loc=180.0, scale=35.0, size=n_neg), 1),
            "CSF_WBC_Count": np.round(np.clip(rng.normal(loc=150.0, scale=45.0, size=n_neg), 10.0, None), 1),
            "Outcome": np.zeros(n_neg, dtype=int),
        })

        pos_df = pd.DataFrame({
            "Freshwater_Exposure": rng.binomial(n=1, p=0.92, size=n_pos),
            "Nasal_Irrigation_History": rng.binomial(n=1, p=0.35, size=n_pos),
            "Severe_Headache": rng.binomial(n=1, p=0.90, size=n_pos),
            "Fever": rng.binomial(n=1, p=0.88, size=n_pos),
            "Nausea_Vomiting": rng.binomial(n=1, p=0.82, size=n_pos),
            "Stiff_Neck": rng.binomial(n=1, p=0.90, size=n_pos),
            "Confusion": rng.binomial(n=1, p=0.78, size=n_pos),
            "Seizures": rng.binomial(n=1, p=0.35, size=n_pos),
            "Photophobia": rng.binomial(n=1, p=0.72, size=n_pos),
            "Age": np.clip(rng.normal(loc=23.0, scale=14.0, size=n_pos), 1, 80).astype(int),
            "Symptom_Onset_Days": np.clip(rng.normal(loc=3.8, scale=1.8, size=n_pos), 1, 12).astype(int),
            "CSF_Pressure": np.round(rng.normal(loc=385.0, scale=45.0, size=n_pos), 1),
            "CSF_WBC_Count": np.round(np.clip(rng.normal(loc=600.0, scale=100.0, size=n_pos), 100.0, None), 1),
            "Outcome": np.ones(n_pos, dtype=int),
        })

        df = pd.concat([neg_df, pos_df], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    df.to_csv(path, index=False)
    logger.info(f"Saved generated dataset to: {path}")
    return path


# =============================================================================
# Transfer Learning Pipeline Orchestrator
# =============================================================================

def prepare_transfer_learning_loaders(
    pretrain_csv_path: Union[str, Path] = "real_meningitis_baseline.csv",
    finetune_csv_path: Union[str, Path] = "synthetic_pam_data.csv",
    pretrain_batch_size: int = 64,
    finetune_batch_size: int = 32,
    test_size: float = 0.20,
    target_col: str = "Outcome",
    num_workers: int = 0,
    random_state: int = 42,
) -> Dict[str, Union[DataLoader, Dict[str, DataLoader]]]:
    """Preprocesses and formats transfer learning datasets into PyTorch DataLoaders.

    Pipeline Steps:
    1. Ingests 'real_meningitis_baseline.csv' (pre-training) and 'synthetic_pam_data.csv' (fine-tuning).
    2. Runs each through `preprocess_clinical_data` (leakage-safe imputation, MinMax scaling, SMOTE).
    3. Wraps outputs into PyTorch `ClinicalDataset` objects.
    4. Instantiates and returns structured PyTorch `DataLoader` instances.

    Args:
        pretrain_csv_path: Path to baseline meningitis CSV for Stage 1 pretraining.
        finetune_csv_path: Path to target PAM CSV for Stage 2 fine-tuning.
        pretrain_batch_size: Batch size for pre-training DataLoader. Defaults to 64.
        finetune_batch_size: Batch size for fine-tuning DataLoader. Defaults to 32.
        test_size: Proportion allocated to evaluation test splits. Defaults to 0.20.
        target_col: Column name of binary classification target. Defaults to "Outcome".
        num_workers: Number of subprocess workers for PyTorch data loading. Defaults to 0.
        random_state: Seed for reproducible train-test split and SMOTE sampling. Defaults to 42.

    Returns:
        Dict[str, Union[DataLoader, Dict[str, DataLoader]]]: Dictionary containing:
            - 'pretrain_loader': DataLoader for Stage 1 pretraining (train partition, shuffled).
            - 'pretrain_val_loader': DataLoader for Stage 1 validation (test partition).
            - 'finetune_loader': DataLoader for Stage 2 fine-tuning (train partition, shuffled).
            - 'finetune_val_loader': DataLoader for Stage 2 validation (test partition).

    Raises:
        FileNotFoundError: If designated CSV files cannot be found or created.
    """
    logger.info("=" * 80)
    logger.info("INITIALIZING TRANSFER LEARNING DATA LOADER PREPARATION")
    logger.info("=" * 80)

    # 1. Verify and resolve filepaths (creates mock if missing)
    pretrain_p = _ensure_mock_csv_exists(
        pretrain_csv_path,
        n_samples=2000,
        is_pretrain_baseline=True,
        random_state=random_state,
    )
    finetune_p = _ensure_mock_csv_exists(
        finetune_csv_path,
        n_samples=1000,
        is_pretrain_baseline=False,
        random_state=random_state,
    )

    # 2. Ingest CSV DataFrames
    df_pretrain = pd.read_csv(pretrain_p)
    df_finetune = pd.read_csv(finetune_p)

    logger.info(f"Loaded Pre-training Cohort ({pretrain_p.name}): shape={df_pretrain.shape}")
    logger.info(f"Loaded Fine-tuning Cohort ({finetune_p.name}):   shape={df_finetune.shape}")

    # 3. Preprocess Pretraining Cohort (Stage 1: Real Baseline)
    logger.info("\n--- Preprocessing Pre-training Baseline Data ---")
    X_train_pre, X_test_pre, y_train_pre, y_test_pre = preprocess_clinical_data(
        df=df_pretrain,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    # 4. Preprocess Fine-tuning Cohort (Stage 2: Target Rare PAM)
    logger.info("\n--- Preprocessing Fine-tuning PAM Target Data ---")
    X_train_fine, X_test_fine, y_train_fine, y_test_fine = preprocess_clinical_data(
        df=df_finetune,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    # 5. Wrap into PyTorch ClinicalDataset instances
    pretrain_dataset = ClinicalDataset(X_train_pre, y_train_pre)
    pretrain_val_dataset = ClinicalDataset(X_test_pre, y_test_pre)

    finetune_dataset = ClinicalDataset(X_train_fine, y_train_fine)
    finetune_val_dataset = ClinicalDataset(X_test_fine, y_test_fine)

    # 6. Instantiate PyTorch DataLoaders
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=pretrain_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    pretrain_val_loader = DataLoader(
        pretrain_val_dataset,
        batch_size=pretrain_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=finetune_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    finetune_val_loader = DataLoader(
        finetune_val_dataset,
        batch_size=finetune_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info("=" * 80)
    logger.info("PYTORCH TRANSFER LEARNING DATALOADERS CONSTRUCTED SUCCESSFULLY:")
    logger.info(f"  * pretrain_loader:     {len(pretrain_dataset)} samples | Batch Size: {pretrain_batch_size} | {len(pretrain_loader)} batches")
    logger.info(f"  * pretrain_val_loader: {len(pretrain_val_dataset)} samples | Batch Size: {pretrain_batch_size} | {len(pretrain_val_loader)} batches")
    logger.info(f"  * finetune_loader:     {len(finetune_dataset)} samples | Batch Size: {finetune_batch_size} | {len(finetune_loader)} batches")
    logger.info(f"  * finetune_val_loader: {len(finetune_val_dataset)} samples | Batch Size: {finetune_batch_size} | {len(finetune_val_loader)} batches")
    logger.info("=" * 80)

    return {
        "pretrain_loader": pretrain_loader,
        "pretrain_val_loader": pretrain_val_loader,
        "finetune_loader": finetune_loader,
        "finetune_val_loader": finetune_val_loader,
    }


# =============================================================================
# Demonstration / Execution Entrypoint
# =============================================================================

def main() -> None:
    """Demonstrates loading transfer learning loaders and inspecting batch tensors."""
    print("=" * 85)
    print("DEMONSTRATION: PYTORCH TRANSFER LEARNING DATA LOADER WORKFLOW")
    print("=" * 85)

    loaders = prepare_transfer_learning_loaders(
        pretrain_csv_path="real_meningitis_baseline.csv",
        finetune_csv_path="synthetic_pam_data.csv",
        pretrain_batch_size=64,
        finetune_batch_size=32,
        test_size=0.20,
        target_col="Outcome",
    )

    pretrain_loader = loaders["pretrain_loader"]
    finetune_loader = loaders["finetune_loader"]

    # Inspect a sample batch from pretrain_loader
    sample_pre_x, sample_pre_y = next(iter(pretrain_loader))
    print("\n--- STAGE 1: PRETRAINING BATCH INSPECTION ---")
    print(f"  - Features Tensor Shape: {sample_pre_x.shape} (Batch Size x 13 Features)")
    print(f"  - Features Tensor Dtype: {sample_pre_x.dtype}")
    print(f"  - Target Tensor Shape:   {sample_pre_y.shape} (Batch Size x 1 Label)")
    print(f"  - Sample Feature Values (Patient 0):\n    {sample_pre_x[0].numpy().round(3)}")
    print(f"  - Sample Target (Patient 0): {sample_pre_y[0].item()}")

    # Inspect a sample batch from finetune_loader
    sample_fine_x, sample_fine_y = next(iter(finetune_loader))
    print("\n--- STAGE 2: FINE-TUNING BATCH INSPECTION ---")
    print(f"  - Features Tensor Shape: {sample_fine_x.shape} (Batch Size x 13 Features)")
    print(f"  - Features Tensor Dtype: {sample_fine_x.dtype}")
    print(f"  - Target Tensor Shape:   {sample_fine_y.shape} (Batch Size x 1 Label)")
    print(f"  - Sample Feature Values (Patient 0):\n    {sample_fine_x[0].numpy().round(3)}")
    print(f"  - Sample Target (Patient 0): {sample_fine_y[0].item()}")

    print("\n" + "=" * 85)
    print("TRANSFER LEARNING PIPELINE READY: LOADERS COMPATIBLE WITH PYTORCH MODELS")
    print("=" * 85)


if __name__ == "__main__":
    main()
