#!/usr/bin/env python
"""Run the training-paradigm substrate builder from the project root.

The real implementation lives in training/tools/.  This launcher keeps
relative --pbf paths such as data/china/guangdong-260918.osm.pbf working
regardless of the caller's current directory.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training"

# Make root-relative data paths work even though the builder runs in training/.
for i, arg in enumerate(sys.argv[:-1]):
    if arg in ("--pbf", "--clean-dir", "--out"):
        p = Path(sys.argv[i + 1])
        if not p.is_absolute():
            in_train = TRAIN / p
            in_root = ROOT / p
            if in_train.exists() or not in_root.exists():
                sys.argv[i + 1] = str(in_train)
            else:
                sys.argv[i + 1] = str(in_root)

os.chdir(TRAIN)
sys.path.insert(0, str(TRAIN))
runpy.run_path(str(TRAIN / "tools" / "build_legacy_substrate.py"),
               run_name="__main__")
