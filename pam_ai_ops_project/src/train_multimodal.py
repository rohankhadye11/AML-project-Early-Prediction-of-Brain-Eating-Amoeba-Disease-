"""Training Pipeline for Multimodal Late Fusion Neural Network in PyTorch.

Executes end-to-end training and evaluation:
1. Generates 500 multimodal patient records (9-D clinical features + 224x224 grayscale MRI slices).
2. Performs stratified 80/20 train-validation splitting.
3. Configures PyTorch DataLoaders with image augmentation for training.
4. Trains PAMLateFusionNet using Adam (lr=1e-4) and BCEWithLogitsLoss (pos_weight=2.0).
5. Tracks and logs validation loss, Accuracy, Recall, and ROC-AUC per epoch.
6. Persists the top-performing model weights to 'best_multimodal_model.pth'.
"""

from typing import Optional, Tuple
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add local src/ directory to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from multimodal_dataset import PAMMultimodalDataset
from multimodal_model import PAMLateFusionNet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def generate_mock_multimodal_dataset(
    n_samples: int = 500,
    pam_ratio: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic multimodal cohort for immediate training verification.

    Simulates:
    - 9 RFE-optimized clinical features (scaled [0.0, 1.0]).
    - 224x224 grayscale brain MRI scans with subtle hyperintense basal lesions.
    - Binary labels (0 = Control, 1 = PAM Positive).

    Args:
        n_samples: Total number of multimodal records to simulate. Defaults to 500.
        pam_ratio: Proportion of positive PAM cases. Defaults to 0.20.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - clinical_data: Shape (500, 9), float32.
            - images: Shape (500, 224, 224), uint8.
            - labels: Shape (500,), int64.
    """
    rng = np.random.default_rng(random_state)
    n_pos = int(round(n_samples * pam_ratio))
    n_neg = n_samples - n_pos

    # 1. Generate 9 Clinical Features:
    # 6 Binary: Freshwater_Exposure, Severe_Headache, Fever, Nausea, Stiff_Neck, Photophobia
    # 3 Continuous: Age, Symptom_Onset_Days, CSF_Pressure (normalized [0, 1])
    neg_clinical = np.column_stack([
        rng.binomial(1, 0.15, size=n_neg),  # Freshwater
        rng.binomial(1, 0.45, size=n_neg),  # Headache
        rng.binomial(1, 0.50, size=n_neg),  # Fever
        rng.binomial(1, 0.40, size=n_neg),  # Nausea
        rng.binomial(1, 0.25, size=n_neg),  # Stiff Neck
        rng.binomial(1, 0.25, size=n_neg),  # Photophobia
        rng.uniform(0.1, 0.9, size=n_neg),  # Age (normalized)
        rng.uniform(0.2, 0.8, size=n_neg),  # Onset days (normalized)
        rng.uniform(0.1, 0.5, size=n_neg),  # CSF Pressure (normalized standard)
    ])

    pos_clinical = np.column_stack([
        rng.binomial(1, 0.92, size=n_pos),  # Freshwater (strong exposure)
        rng.binomial(1, 0.90, size=n_pos),  # Severe headache
        rng.binomial(1, 0.88, size=n_pos),  # High fever
        rng.binomial(1, 0.82, size=n_pos),  # Nausea
        rng.binomial(1, 0.90, size=n_pos),  # Meningismus
        rng.binomial(1, 0.75, size=n_pos),  # Photophobia
        rng.uniform(0.05, 0.4, size=n_pos), # Age (skewed younger)
        rng.uniform(0.1, 0.4, size=n_pos),  # Rapid onset
        rng.uniform(0.6, 0.95, size=n_pos), # Critical CSF Pressure
    ])

    clinical_data = np.vstack([neg_clinical, pos_clinical]).astype(np.float32)
    labels = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])

    # 2. Generate 224x224 Grayscale MRI Images:
    images = np.zeros((n_samples, 224, 224), dtype=np.uint8)
    for i in range(n_samples):
        img = np.zeros((224, 224), dtype=np.uint8)
        # Brain parenchyma ellipse
        cv2.ellipse(img, (112, 112), (75, 90), 0, 0, 360, 100, -1)
        # Sulcal folds
        for ang in range(0, 360, 30):
            rad = np.deg2rad(ang)
            cv2.line(img, (int(112 + 65*np.cos(rad)), int(112 + 80*np.sin(rad))),
                          (int(112 + 50*np.cos(rad)), int(112 + 65*np.sin(rad))), 60, 1)
        # Ventricles
        cv2.circle(img, (102, 112), 10, 30, -1)
        cv2.circle(img, (122, 112), 10, 30, -1)

        # Pathological edema marker for PAM positive cases
        if labels[i] == 1:
            cv2.circle(img, (95, 88), 16, 210, -1)  # Hyperintense edema focus
            img = cv2.GaussianBlur(img, (7, 7), 2.0)
        else:
            img = cv2.GaussianBlur(img, (5, 5), 1.0)

        images[i] = img

    # Shuffle synchronously
    indices = rng.permutation(n_samples)
    return clinical_data[indices], images[indices], labels[indices]


def train_multimodal_model(
    n_samples: int = 500,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    pos_weight_val: float = 2.0,
    weights_save_path: str = "best_multimodal_model.pth",
    device: Optional[str] = None,
) -> PAMLateFusionNet:
    """Trains and evaluates the PAMLateFusionNet model over 10 epochs.

    Args:
        n_samples: Size of simulated cohort. Defaults to 500.
        batch_size: Batch size for mini-batch gradient descent. Defaults to 32.
        num_epochs: Number of training epochs. Defaults to 10.
        learning_rate: Optimizer learning rate. Defaults to 1e-4.
        pos_weight_val: Loss weight for positive class in BCEWithLogitsLoss. Defaults to 2.0.
        weights_save_path: Path to save top model checkpoint. Defaults to 'best_multimodal_model.pth'.
        device: 'cuda' or 'cpu'. If None, autodetects.

    Returns:
        PAMLateFusionNet: Trained PyTorch neural network.
    """
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using execution device: {selected_device}")

    # Step 1: Generate Mock Data
    logger.info(f"Generating mock multimodal dataset ({n_samples} samples)...")
    clinical_arr, image_arr, label_arr = generate_mock_multimodal_dataset(n_samples=n_samples, random_state=42)

    # Step 2: Stratified 80/20 Train-Validation Split
    train_idx, val_idx = train_test_split(
        np.arange(n_samples),
        test_size=0.20,
        stratify=label_arr,
        random_state=42,
    )

    train_clin, val_clin = clinical_arr[train_idx], clinical_arr[val_idx]
    train_imgs, val_imgs = image_arr[train_idx], image_arr[val_idx]
    train_lbls, val_lbls = label_arr[train_idx], label_arr[val_idx]

    logger.info(f"Dataset split: Train = {len(train_idx)} samples | Validation = {len(val_idx)} samples")

    # Step 3: PyTorch Datasets and DataLoaders
    train_dataset = PAMMultimodalDataset(
        clinical_data=train_clin,
        images=train_imgs,
        labels=train_lbls,
        is_training=True,  # Augmentations enabled
    )
    val_dataset = PAMMultimodalDataset(
        clinical_data=val_clin,
        images=val_imgs,
        labels=val_lbls,
        is_training=False, # Deterministic validation
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 4: Model Initialization
    model = PAMLateFusionNet(
        num_clinical_features=9,
        image_embedding_dim=64,
        clinical_embedding_dim=32,
        pretrained=True,
    ).to(selected_device)

    # Step 5: Weighted Loss Function & Optimizer
    # pos_weight penalizes False Negatives (fatal in PAM)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(selected_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Step 6: Training Loop
    logger.info("=" * 85)
    logger.info(f"STARTING MULTIMODAL LATE FUSION TRAINING ({num_epochs} EPOCHS)")
    logger.info("=" * 85)

    best_val_roc = 0.0
    best_epoch = 0

    print(f"{'Epoch':8} | {'Train Loss':11} | {'Val Loss':10} | {'Val Acc':9} | {'Val Recall':11} | {'Val ROC-AUC':12} | Status")
    print("-" * 85)

    for epoch in range(1, num_epochs + 1):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0

        for images_b, clinical_b, labels_b in train_loader:
            images_b = images_b.to(selected_device)
            clinical_b = clinical_b.to(selected_device)
            labels_b = labels_b.to(selected_device)

            optimizer.zero_grad()
            logits = model(images_b, clinical_b)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(labels_b)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_preds_list = []
        val_probs_list = []
        val_targets_list = []

        with torch.no_grad():
            for images_b, clinical_b, labels_b in val_loader:
                images_b = images_b.to(selected_device)
                clinical_b = clinical_b.to(selected_device)
                labels_b = labels_b.to(selected_device)

                logits = model(images_b, clinical_b)
                loss = criterion(logits, labels_b)
                running_val_loss += loss.item() * len(labels_b)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.50).astype(int)

                val_probs_list.extend(probs)
                val_preds_list.extend(preds)
                val_targets_list.extend(labels_b.cpu().numpy().flatten())

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_acc = accuracy_score(val_targets_list, val_preds_list)
        val_rec = recall_score(val_targets_list, val_preds_list, zero_division=0)
        val_roc = roc_auc_score(val_targets_list, val_probs_list)

        # Checkpoint Saving: Best ROC-AUC
        status_msg = ""
        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_epoch = epoch
            torch.save(model.state_dict(), weights_save_path)
            status_msg = f"* Saved ({weights_save_path})"

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"{epoch_train_loss:10.4f} | "
            f"{epoch_val_loss:9.4f} | "
            f"{val_acc*100:7.2f}% | "
            f"{val_rec*100:9.2f}% | "
            f"{val_roc:11.4f} | "
            f"{status_msg}"
        )

    print("=" * 85)
    logger.info(f"Training completed. Best model checkpoint from Epoch {best_epoch:02d} with Val ROC-AUC = {best_val_roc:.4f}")
    logger.info(f"Model weights successfully saved to '{weights_save_path}'.")

    # Step 7: Intermediate Embedding Verification
    sample_img, sample_clin, _ = val_dataset[0]
    sample_img = sample_img.unsqueeze(0).to(selected_device)      # (1, 1, 224, 224)
    sample_clin = sample_clin.unsqueeze(0).to(selected_device)    # (1, 9)

    model.eval()
    with torch.no_grad():
        img_emb, clin_emb, fused_emb = model.get_embeddings(sample_img, sample_clin)

    print("\n--- LATENT EMBEDDING VERIFICATION (FOR t-SNE / EXPLAINABILITY) ---")
    print(f"  * Neuroimaging Embedding Shape: {tuple(img_emb.shape)}  <-- Exact 64-D Latent Vector")
    print(f"  * Clinical Tabular Embedding:  {tuple(clin_emb.shape)}  <-- Exact 32-D Latent Vector")
    print(f"  * Fused Latent Representation: {tuple(fused_emb.shape)}  <-- Exact 96-D Concatenated Vector")
    print("=" * 85)

    return model


def main() -> None:
    """Entry point for executing multimodal neural network training."""
    train_multimodal_model(
        n_samples=500,
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-4,
        pos_weight_val=2.0,
        weights_save_path="best_multimodal_model.pth",
    )


if __name__ == "__main__":
    main()
