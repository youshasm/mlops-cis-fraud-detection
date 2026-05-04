"""Pytest configuration: ensure repo root is in sys.path."""
import sys
from pathlib import Path

# Add repo root to sys.path so local packages can be imported
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
