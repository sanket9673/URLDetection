"""
Global pytest configuration and environment setup for URLDetection test suite.
Ensures proper C runtime initialization order between LightGBM and PyTorch on macOS.
"""

import os

# Prevent OpenMP runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Pre-initialize LightGBM C library before any dynamic PyTorch loads
try:
    import lightgbm
except ImportError:
    pass
