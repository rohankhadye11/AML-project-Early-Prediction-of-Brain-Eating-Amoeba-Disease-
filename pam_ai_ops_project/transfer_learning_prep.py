"""Root entrypoint for Transfer Learning Data Preparation."""

import sys
from pathlib import Path

# Add src/ to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from transfer_learning_prep import (
    ClinicalDataset,
    prepare_transfer_learning_loaders,
    main,
)

if __name__ == "__main__":
    main()
