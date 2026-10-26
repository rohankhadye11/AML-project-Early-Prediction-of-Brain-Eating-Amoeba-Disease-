"""Root entrypoint for Phase 5: Medical Explainable AI (XAI) Execution Pipeline."""

import sys
from pathlib import Path

# Add src/ to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from run_phase5_xai import main

if __name__ == "__main__":
    main()
