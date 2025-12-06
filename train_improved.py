#!/usr/bin/env python3
"""
Improved Training Script for Skin Cancer Classification
- Data augmentation
- Learning rate scheduling
- Early stopping
- Class weights for balance
- Fine-tuning strategy
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SKIN CANCER CLASSIFICATION - IMPROVED TRAINING SCRIPT")
print("=" * 80)

# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from pathlib import Path

print(f"\n✅ TensorFlow {tf.__version__}")
print(f"✅ NumPy {np.__version__}")
print(f"✅ GPU Available: {tf.config.list_physical_devices('GPU')}")

# Configuration
CONFIG = {
    'data_dir': 'data/train',
    'test_dir': 'data/test',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs_phase1': 20,      # Frozen base
    'epochs_phase2': 20,      # Fine-tuning
    'learning_rate_phase1': 0.001,
    'learning_rate_phase2': 0.0001,
    'seed': 42,
    'model_save_path': 'model/skin_cancer_model_improved.keras'
}

# Set random seeds
np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])

# Data Loading
print("\n" + "="*80)
print("PHASE 1: DATA LOADING")
print("="*80)

def count_images(directory):
    """Count images in directory"""
    benign = len(list(Path(directory).glob('benign/*.jpg'))) + \
             len(list(Path(directory).glob('benign/*.jpeg'))) + \
             len(list(Path(directory).glob('benign/*.png')))
    malignant = len(list(Path(directory).glob('malignant/*.jpg'))) + \
                len(list(Path(directory).glob('malignant/*.jpeg'))) + \
                len(list(Path(directory).glob('malignant/*.png')))
    return benign, malignant

train_benign, train_malignant = count_images(CONFIG['data_dir'])
test_benign, test_malignant = count_images(CONFIG['test_dir'])

print(f"\n📊 Dataset Statistics:")
print(f"   Train: {train_benign} benign, {train_malignant} malignant = {train_benign + train_malignant} total")
print(f"   Test:  {test_benign} benign, {test_malignant} malignant = {test_benign + test_malignant} total")
print(f"   Class ratio: {train_benign/train_malignant:.2f}:1 (benign:malignant)")

# Calculate class weights for imbalanced data
total_train = train_benign + train_malignant
weight_for_benign = (1 / train_benign) * (total_train / 2.0)
weight_for_malignant = (1 / train_malignant) * (total_train / 2.0)
class_weight = {0: weight_for_benign, 1: weight_for_malignant}

print(f"\n⚖️ Class Weights (to balance dataset):")
print(f"   Benign (0): {weight_for_benign:.4f}")
print(f"   Malignant (1): {weight_for_malignant:.4f}")

# Load datasets
print("\n📂 Loading datasets...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    CONFIG['data_dir'],
    seed=CONFIG['seed'],
    image_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    label_mode='binary',
    class_names=['benign', 'malignant']
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    CONFIG['test_dir'],
    seed=CONFIG['seed'],
    image_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    label_mode='binary',
    class_names=['benign', 'malignant']
)

# Split train into train/val
val_split = 0.2
train_size = int(len(train_ds) * (1 - val_split))
train_ds_final = train_ds.take(train_size)
val_ds = train_ds.skip(train_size)

print(f"✅ Train batches: {train_size}")
print(f"✅ Validation batches: {len(val_ds)}")
print(f"✅ Test batches: {len(test_ds)}")

# Data Augmentation
print("\n" + "="*80)
print("PHASE 2: DATA AUGMENTATION")
print("="*80)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")

print("✅ Augmentation layers:")
print("   - Random flip (horizontal & vertical)")
print("   - Random rotation (±30%)")
print("   - Random zoom (±20%)")
print("   - Random brightness (±20%)")
print("   - Random contrast (±20%)")

# Normalization
normalization_layer = layers.Rescaling(1./127.5, offset=-1)

# Apply augmentation and normalization
def prepare_dataset(ds, augment=False):
    """Prepare dataset with augmentation and normalization"""
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds_prepared = prepare_dataset(train_ds_final, augment=True)
val_ds_prepared = prepare_dataset(val_ds, augment=False)
test_ds_prepared = prepare_dataset(test_ds, augment=False)

print("✅ Datasets prepared with augmentation")

# Model Building
print("\n" + "="*80)
print("PHASE 3: MODEL BUILDING")
print("="*80)

def build_model(trainable_base=False, trainable_layers=0):
    """Build MobileNetV2 model"""
    
    # Base model
    base_model = keras.applications.MobileNetV2(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze/unfreeze base
    base_model.trainable = trainable_base
    if trainable_base and trainable_layers > 0:
        # Unfreeze only last N layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    
    # Build full model
    inputs = keras.Input(shape=(*CONFIG['img_size'], 3))
    x = inputs
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Increased dropout
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model

model = build_model(trainable_base=False)

print(f"✅ Model built: MobileNetV2 + Custom Head")
print(f"   Input shape: {CONFIG['img_size'] + (3,)}")
print(f"   Output: Binary classification (sigmoid)")
print(f"   Total parameters: {model.count_params():,}")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate_phase1']),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

print(f"✅ Model compiled")
print(f"   Optimizer: Adam (lr={CONFIG['learning_rate_phase1']})")
print(f"   Loss: Binary crossentropy")
print(f"   Metrics: Accuracy, Precision, Recall, AUC")

# Callbacks
print("\n" + "="*80)
print("PHASE 4: CALLBACKS")
print("="*80)

callbacks = [
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Learning rate reduction
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Model checkpoint
    keras.callbacks.ModelCheckpoint(
        CONFIG['model_save_path'],
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    
    # CSV logger
    keras.callbacks.CSVLogger('training_log.csv', append=True)
]

print("✅ Callbacks configured:")
print("   - Early stopping (patience=5)")
print("   - Learning rate reduction (patience=3, factor=0.5)")
print("   - Model checkpoint (save best)")
print("   - CSV logger")

# Training Phase 1: Frozen Base
print("\n" + "="*80)
print("PHASE 5: TRAINING (Phase 1 - Frozen Base)")
print("="*80)

print("\n🚀 Starting training with frozen MobileNetV2 base...")
print(f"   Epochs: {CONFIG['epochs_phase1']}")
print(f"   Learning rate: {CONFIG['learning_rate_phase1']}")

history_phase1 = model.fit(
    train_ds_prepared,
    validation_data=val_ds_prepared,
    epochs=CONFIG['epochs_phase1'],
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

print("\n✅ Phase 1 training completed!")

# Training Phase 2: Fine-tuning
print("\n" + "="*80)
print("PHASE 6: FINE-TUNING (Phase 2 - Unfreeze Last Layers)")
print("="*80)

# Unfreeze last 30 layers
base_model = model.layers[1]
base_model.trainable = True

trainable_layers = 30
for layer in base_model.layers[:-trainable_layers]:
    layer.trainable = False

print(f"🔓 Unfrozen last {trainable_layers} layers of MobileNetV2")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate_phase2']),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

print(f"✅ Recompiled with lower learning rate: {CONFIG['learning_rate_phase2']}")

print(f"\n🚀 Starting fine-tuning...")
print(f"   Additional epochs: {CONFIG['epochs_phase2']}")

history_phase2 = model.fit(
    train_ds_prepared,
    validation_data=val_ds_prepared,
    epochs=CONFIG['epochs_phase2'],
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

print("\n✅ Phase 2 fine-tuning completed!")

# Evaluation
print("\n" + "="*80)
print("PHASE 7: EVALUATION")
print("="*80)

print("\n📊 Evaluating on test set...")
test_results = model.evaluate(test_ds_prepared, verbose=1)

print(f"\n🎯 Final Test Results:")
print(f"   Loss: {test_results[0]:.4f}")
print(f"   Accuracy: {test_results[1]*100:.2f}%")
print(f"   Precision: {test_results[2]*100:.2f}%")
print(f"   Recall: {test_results[3]*100:.2f}%")
print(f"   AUC: {test_results[4]:.4f}")

# Save final model
final_model_path = 'model/skin_cancer_model_final.keras'
model.save(final_model_path)
print(f"\n💾 Model saved to: {final_model_path}")

# Plot training history
print("\n" + "="*80)
print("PHASE 8: VISUALIZATION")
print("="*80)

def plot_history(history1, history2):
    """Plot training history"""
    
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=150)
    print("✅ Training history plot saved to: training_history_improved.png")
    
    try:
        plt.show()
    except:
        pass

plot_history(history_phase1, history_phase2)

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\n📁 Files generated:")
print(f"   - {final_model_path}")
print(f"   - training_history_improved.png")
print(f"   - training_log.csv")
print(f"\n🎉 You can now use the improved model for better predictions!")
print("="*80)
