"""``python -m src.drawing INPUT OUT_DIR [options]`` -> ``cli.main`` (see ``cli.py``)."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
