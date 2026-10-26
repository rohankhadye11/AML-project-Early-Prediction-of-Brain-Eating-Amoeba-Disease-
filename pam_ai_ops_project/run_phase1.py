"""Root runner for Phase 1: Robust Data Engineering & Sensitivity Analysis."""

import sys
from pathlib import Path

# Add src/ to Python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from run_phase1 import main

if __name__ == "__main__":
    main()
