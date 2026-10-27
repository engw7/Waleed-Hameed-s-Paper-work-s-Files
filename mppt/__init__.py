"""Library behind run_all.py and experiments/*.py (ported from the v3 notebook cells; see docs/METRICS.md for metric definitions)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')     # identical floating-point order on every run
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
