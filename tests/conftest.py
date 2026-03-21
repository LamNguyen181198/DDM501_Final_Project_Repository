"""Shared pytest configuration — ensures sys.path is correct for all tests.

Root-level shims (data_ingestion.py, feature_engineering.py) must be resolved
before the pipeline-internal shim (pipeline/feature_engineering.py) so that
absolute imports in train.py and feature_engineer.py resolve correctly.

We remove-then-reinsert both paths to guarantee ordering even when pytest has
already added repo_root before conftest runs.
Final order: [repo_root, pipeline_dir, ...]
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent

# Insert in reverse of desired final order: first pipeline (goes to index 0),
# then repo_root (also goes to index 0, pushing pipeline to index 1).
# Remove any existing entry first to avoid duplicates at wrong positions.
for _path in [str(_repo_root / "pipeline"), str(_repo_root)]:
    while _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
