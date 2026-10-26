"""Root entrypoint for Multimodal Late Fusion Neural Network Training."""

import sys
from pathlib import Path

# Add src/ to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from train_multimodal import main

if __name__ == "__main__":
    main()
