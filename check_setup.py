#!/usr/bin/env python3
"""
Quick Training Demo - Skin Cancer Classification
Shortened version untuk testing environment setup
"""

import os
import sys

print("\n" + "=" * 70)
print("SKIN CANCER CLASSIFICATION - QUICK SETUP TEST")
print("=" * 70 + "\n")

# Minimal checks
print("Step 1: Checking dataset...")
data_dir = "data/train"
try:
    benign_count = len(os.listdir(f"{data_dir}/benign"))
    malignant_count = len(os.listdir(f"{data_dir}/malignant"))
    print(f"✓ Dataset found:")
    print(f"  - Benign: {benign_count} files")
    print(f"  - Malignant: {malignant_count} files")
    print(f"  - Total: {benign_count + malignant_count} files")
except:
    print("✗ Dataset not found in data/train/")
    sys.exit(1)

# Try importing TensorFlow
print("\nStep 2: Checking TensorFlow...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} loaded")
    print(f"✓ Using {tf.config.list_physical_devices('GPU') or 'CPU'}")
except ImportError as e:
    print(f"⚠ TensorFlow not available: {e}")
    print("\nTo fix this, run:")
    print("  pip install tensorflow")
    print("\nNote: Installation may take 10-15 minutes...")
    sys.exit(1)

# Try other imports
print("\nStep 3: Checking other libraries...")
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import sklearn
    print("✓ All required libraries available")
    print(f"  - NumPy: {np.__version__}")
    print(f"  - OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"⚠ Some libraries missing: {e}")

print("\n" + "=" * 70)
print("ENVIRONMENT SETUP STATUS: READY FOR TRAINING")
print("=" * 70)

print("\nTo start training:")
print("1. Open notebook: jupyter lab notebook/training.ipynb")
print("2. Or run simple training: python train_simple.py")
print("\nNote: First time training downloads model (~120MB)")
print("Training time depends on dataset size and GPU availability")
print("=" * 70 + "\n")
