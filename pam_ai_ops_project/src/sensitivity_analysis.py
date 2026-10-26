"""Clinical Sensitivity Analysis and Noise Injection Module.

This module provides the ClinicalNoiseInjector class for stress-testing model
robustness against real-world hospital data imperfections, including clerical
data-entry errors, measurement variances, and neuroimaging artifacts.
"""

from typing import List, Optional, Tuple, Union
import logging
import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClinicalNoiseInjector:
    """Simulates real-world clinical data corruptions for sensitivity analysis.

    Provides methods to simulate:
    1. Measurement variance in continuous biomarkers (e.g., sensor calibration drift).
    2. Human clerical errors in binary intake questionnaires (e.g., miskeyed checkboxes).
    3. Motion blur and scanner artifacts in neuroimaging MRI scans.

    Attributes:
        random_state: Optional seed for reproducible perturbation testing.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        """Initializes the ClinicalNoiseInjector.

        Args:
            random_state: Seed for pseudo-random number generator. Defaults to None.
        """
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def perturb_continuous(
        self,
        df: pd.DataFrame,
        columns: List[str],
        variance: float = 0.10,
        subset_fraction: float = 1.0,
        clip_bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
    ) -> pd.DataFrame:
        """Randomly alters continuous features by up to ±variance.

        Simulates physiologic variability and clinical instrument measurement drift
        by applying multiplicative uniform perturbation:
            x_perturbed = x * (1 + delta), where delta ~ Uniform(-variance, +variance)

        Args:
            df: Input pandas DataFrame containing continuous clinical features.
            columns: List of continuous column names to perturb.
            variance: Maximum relative perturbation fraction (e.g., 0.10 for ±10%).
                Defaults to 0.10.
            subset_fraction: Fraction of rows to perturb (1.0 = all rows).
                Defaults to 1.0.
            clip_bounds: Optional (min, max) tuple to constrain perturbed values
                within valid ranges (e.g., [0.0, 1.0] for scaled data). Defaults to (0.0, 1.0).

        Returns:
            pd.DataFrame: A new DataFrame with perturbed continuous features.

        Raises:
            ValueError: If variance is negative or columns do not exist in DataFrame.
        """
        if variance < 0.0:
            raise ValueError(f"Variance must be non-negative, received {variance}")

        df_noisy = df.copy()
        existing_cols = [c for c in columns if c in df_noisy.columns]
        missing_cols = set(columns) - set(existing_cols)
        if missing_cols:
            logger.warning(f"Columns not found in DataFrame for continuous perturbation: {missing_cols}")

        if not existing_cols:
            return df_noisy

        n_rows = len(df_noisy)
        if n_rows == 0:
            return df_noisy

        # Select row indices to perturb based on subset_fraction
        if subset_fraction >= 1.0:
            perturbed_indices = df_noisy.index
        else:
            n_subset = max(1, int(round(n_rows * subset_fraction)))
            perturbed_indices = self._rng.choice(df_noisy.index, size=n_subset, replace=False)

        for col in existing_cols:
            col_values = df_noisy.loc[perturbed_indices, col].values.astype(float)
            # Multiplicative noise: (1 + uniform(-variance, variance))
            multipliers = 1.0 + self._rng.uniform(-variance, variance, size=len(col_values))
            noisy_values = col_values * multipliers

            if clip_bounds is not None:
                noisy_values = np.clip(noisy_values, clip_bounds[0], clip_bounds[1])

            df_noisy.loc[perturbed_indices, col] = noisy_values

        logger.info(
            f"Perturbed continuous features {existing_cols} for {len(perturbed_indices)}/{n_rows} "
            f"rows with variance +/-{variance * 100:.1f}%."
        )
        return df_noisy

    def flip_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        flip_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Randomly flips binary indicators to simulate clerical intake errors.

        In emergency intake, medical clerks occasionally invert checklist toggles
        (e.g., misrecording stiff neck or photophobia). This method inverts
        binary flags (0 -> 1, 1 -> 0) with probability `flip_rate`.

        Args:
            df: Input pandas DataFrame containing binary features (0 or 1).
            columns: List of binary column names to corrupt.
            flip_rate: Probability of inverting each individual entry (e.g., 0.05 for 5%).
                Defaults to 0.05.

        Returns:
            pd.DataFrame: A new DataFrame with inverted binary flags.

        Raises:
            ValueError: If flip_rate is not within [0.0, 1.0].
        """
        if not (0.0 <= flip_rate <= 1.0):
            raise ValueError(f"flip_rate must be between 0.0 and 1.0, received {flip_rate}")

        df_noisy = df.copy()
        existing_cols = [c for c in columns if c in df_noisy.columns]
        missing_cols = set(columns) - set(existing_cols)
        if missing_cols:
            logger.warning(f"Columns not found in DataFrame for categorical flipping: {missing_cols}")

        if not existing_cols or flip_rate == 0.0:
            return df_noisy

        total_flipped = 0
        n_rows = len(df_noisy)

        for col in existing_cols:
            mask = self._rng.random(size=n_rows) < flip_rate
            flipped_count = int(np.sum(mask))
            total_flipped += flipped_count
            if flipped_count > 0:
                # Invert: 1 - x (for 0/1 binary flags)
                original_vals = df_noisy.loc[mask, col].values
                df_noisy.loc[mask, col] = 1 - original_vals

        logger.info(
            f"Flipped categorical features {existing_cols} with flip_rate {flip_rate * 100:.1f}%. "
            f"Total bit flips: {total_flipped}."
        )
        return df_noisy

    def add_image_noise(
        self,
        image_array: np.ndarray,
        blur_ksize: Tuple[int, int] = (5, 5),
        noise_ratio: float = 0.02,
    ) -> np.ndarray:
        """Applies Gaussian motion blur and salt-and-pepper noise to an MRI array.

        Simulates patient movement artifacts during neuroimaging acquisition
        and electrical detector transmission noise on 224x224 grayscale MRI scans.

        Args:
            image_array: 2D numpy array representing a grayscale MRI slice (e.g., 224x224).
            blur_ksize: Kernel size for Gaussian blur filter. Must have odd dimensions.
                Defaults to (5, 5).
            noise_ratio: Proportion of image pixels corrupted by salt-and-pepper noise.
                Defaults to 0.02 (2% of pixels).

        Returns:
            np.ndarray: Corrupted 224x224 grayscale image array with same dtype as input.

        Raises:
            ValueError: If input image is not a 2D array or kernel sizes are even.
            TypeError: If input is not a numpy array.
        """
        if not isinstance(image_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, received {type(image_array).__name__}")
        if image_array.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image array, received shape {image_array.shape}")
        if blur_ksize[0] % 2 == 0 or blur_ksize[1] % 2 == 0:
            raise ValueError(f"Kernel dimensions must be odd numbers, received {blur_ksize}")

        is_float = np.issubdtype(image_array.dtype, np.floating)
        orig_dtype = image_array.dtype

        # Step 1: Apply Gaussian Blur (Motion Artifact)
        # OpenCV expects uint8 or float32/float64
        blurred = cv2.GaussianBlur(image_array, blur_ksize, sigmaX=0)

        # Step 2: Apply Salt-and-Pepper Impulse Noise (Detector Noise)
        noisy_image = blurred.copy()
        num_pixels = image_array.size
        num_noise_pixels = int(round(num_pixels * noise_ratio))

        if num_noise_pixels > 0:
            # Salt noise (white pixels)
            num_salt = num_noise_pixels // 2
            salt_coords_y = self._rng.integers(0, image_array.shape[0], size=num_salt)
            salt_coords_x = self._rng.integers(0, image_array.shape[1], size=num_salt)
            salt_val = 1.0 if is_float else 255
            noisy_image[salt_coords_y, salt_coords_x] = salt_val

            # Pepper noise (black pixels)
            num_pepper = num_noise_pixels - num_salt
            pepper_coords_y = self._rng.integers(0, image_array.shape[0], size=num_pepper)
            pepper_coords_x = self._rng.integers(0, image_array.shape[1], size=num_pepper)
            pepper_val = 0.0 if is_float else 0
            noisy_image[pepper_coords_y, pepper_coords_x] = pepper_val

        return noisy_image.astype(orig_dtype)
