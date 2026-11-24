#!/usr/bin/env python
"""Start the local web service."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    a = ap.parse_args()
    uvicorn.run("app.main:app", host=a.host, port=a.port,
                log_level="info", workers=1)


if __name__ == "__main__":
    main()
