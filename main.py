"""Convenience entry point:  python main.py  (same as python -m src.orchestration.run_pipeline)"""
import sys
from src.orchestration.run_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
