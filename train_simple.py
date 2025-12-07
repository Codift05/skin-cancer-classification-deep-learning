#!/usr/bin/env python3
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SKIN CANCER CLASSIFICATION - TRAINING SCRIPT")
print("=" * 70)

print("\n[1/10] Checking environment...")
print(f"Python version: {sys.version}")

print("\n[2/10] Importing libraries...")
try:
    import numpy as np
    print(f"[OK] NumPy {np.__version__}")
except ImportError as e:
    print(f"[FAIL] NumPy: {e}")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    print(f"[OK] TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"[FAIL] TensorFlow: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("[OK] Matplotlib")
except:
    print("[OK] Matplotlib (optional)")

try:
    import cv2
    print(f"[OK] OpenCV {cv2.__version__}")
except:
    print("[WARN] OpenCV (optional)")

try:
    from sklearn.metrics import confusion_matrix, classification_report
    print("[OK] Scikit-learn")
except:
    print("[WARN] Scikit-learn (optional)")

print("\n[3/10] Checking dataset...")
data_dir = "data/train"
if os.path.exists(data_dir):
    benign_files = len([f for f in os.listdir(f"{data_dir}/benign") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    malignant_files = len([f for f in os.listdir(f"{data_dir}/malignant") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[OK] Benign: {benign_files}, Malignant: {malignant_files}, Total: {benign_files + malignant_files}")
else:
    print(f"[FAIL] Data directory not found: {data_dir}")
    sys.exit(1)

print("\n[4/10] Loading dataset...")
try:
    batch_size = 32
    img_size = (224, 224)
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        class_names=['benign', 'malignant']
    )
    print(f"[OK] Dataset loaded")
    
    val_split = 0.2
    train_size = int(len(train_ds) * (1 - val_split))
    train_ds_final = train_ds.take(train_size)
    val_ds = train_ds.skip(train_size)
    print(f"[OK] Train batches: {train_size}, Validation batches: {len(val_ds)}")
    
except Exception as e:
    print(f"[FAIL] Dataset loading: {e}")
    sys.exit(1)

print("\n[5/10] Setting up augmentation...")
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
])
print("[OK] Augmentation ready")

print("\n[6/10] Building model...")
try:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    print("[OK] Model created")
    
except Exception as e:
    print(f"[FAIL] Model building: {e}")
    sys.exit(1)

print("\n[7/10] Compiling model...")
try:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("[OK] Model compiled")
except Exception as e:
    print(f"[FAIL] Compilation: {e}")
    sys.exit(1)

print("\n[8/10] Setting callbacks...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
]
print("[OK] Callbacks ready")

print("\n[9/10] Training model...")
print("=" * 70)
try:
    history = model.fit(
        train_ds_final,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )
    print("=" * 70)
    print("[OK] Training completed")
except Exception as e:
    print(f"[FAIL] Training: {e}")
    sys.exit(1)

print("\n[10/10] Saving model...")
try:
    os.makedirs('model', exist_ok=True)
    model.save('model/skin_cancer_model_optimized_final.keras')
    
    class_names = ['benign', 'malignant']
    with open('model/class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))
    
    print("[OK] Model saved: model/skin_cancer_model_optimized_final.keras")
    print("[OK] Classes saved: model/class_names.txt")
except Exception as e:
    print(f"[FAIL] Saving: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("Next: streamlit run app/app.py")
print("=" * 70)
