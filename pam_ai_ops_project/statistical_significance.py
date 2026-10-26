"""Root entrypoint for Phase 4: Statistical Significance Testing."""

import sys
from pathlib import Path

# Add src/ to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from statistical_significance import main

if __name__ == "__main__":
    main()
