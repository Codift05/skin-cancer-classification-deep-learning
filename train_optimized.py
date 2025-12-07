"""
Optimized Training Script for Skin Cancer Classification
Focus on high accuracy and recall for malignant cases
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Paths
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / 'data' / 'train'
TEST_DIR = BASE_DIR / 'data' / 'test'
MODEL_DIR = BASE_DIR / 'model'
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("SKIN CANCER CLASSIFICATION - OPTIMIZED TRAINING")
print("=" * 70)
print(f"\nTraining Configuration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"\nDataset:")
print(f"  Train Dir: {TRAIN_DIR}")
print(f"  Test Dir: {TEST_DIR}")

# Count images
train_benign = len(list((TRAIN_DIR / 'benign').glob('*')))
train_malignant = len(list((TRAIN_DIR / 'malignant').glob('*')))
test_benign = len(list((TEST_DIR / 'benign').glob('*')))
test_malignant = len(list((TEST_DIR / 'malignant').glob('*')))

print(f"\n  Train: {train_benign} benign, {train_malignant} malignant (Total: {train_benign + train_malignant})")
print(f"  Test:  {test_benign} benign, {test_malignant} malignant (Total: {test_benign + test_malignant})")

# Calculate class weights for imbalanced dataset
total_train = train_benign + train_malignant
weight_benign = total_train / (2 * train_benign)
weight_malignant = total_train / (2 * train_malignant)
class_weights = {0: weight_benign, 1: weight_malignant}

print(f"\nClass Weights (to handle imbalance):")
print(f"  Benign (0): {weight_benign:.4f}")
print(f"  Malignant (1): {weight_malignant:.4f}")

# Aggressive Data Augmentation for Training
print("\n" + "=" * 70)
print("DATA AUGMENTATION")
print("=" * 70)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,           # Increased rotation
    width_shift_range=0.3,       # Increased shift
    height_shift_range=0.3,
    shear_range=0.3,             # Increased shear
    zoom_range=0.3,              # Increased zoom
    horizontal_flip=True,
    vertical_flip=True,          # Added vertical flip
    brightness_range=[0.7, 1.3], # Added brightness variation
    fill_mode='nearest',
    validation_split=0.2         # 20% for validation
)

# Only rescaling for test data
test_datagen = ImageDataGenerator(rescale=1./255)

print("Training data augmentation:")
print("  - Rotation: ±40°")
print("  - Width/Height shift: ±30%")
print("  - Shear: ±30%")
print("  - Zoom: ±30%")
print("  - Horizontal & Vertical flip")
print("  - Brightness: 0.7-1.3x")
print("  - Validation split: 20%")

# Create generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nData generators created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {validation_generator.samples}")
print(f"  Test samples: {test_generator.samples}")
print(f"\nClass mapping: {train_generator.class_indices}")

# Build Model with MobileNetV2
print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

# Load MobileNetV2 base
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze early layers, unfreeze later layers for fine-tuning
base_model.trainable = True
fine_tune_at = 100  # Unfreeze from layer 100 onwards

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Base Model: MobileNetV2 (ImageNet weights)")
print(f"  Total layers: {len(base_model.layers)}")
print(f"  Frozen layers: {fine_tune_at}")
print(f"  Trainable layers: {len(base_model.layers) - fine_tune_at}")

# Build full model
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)  # Increased dropout
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)  # Increased dropout
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

print("\nCustom layers added:")
print("  - GlobalAveragePooling2D")
print("  - BatchNormalization")
print("  - Dropout(0.5)")
print("  - Dense(256) + L2 regularization")
print("  - BatchNormalization")
print("  - Dropout(0.5)")
print("  - Dense(128) + L2 regularization")
print("  - Dropout(0.3)")
print("  - Dense(1, sigmoid)")

# Compile model with optimized metrics
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn')
    ]
)

print("\nModel compiled with:")
print("  Optimizer: Adam (lr=0.001)")
print("  Loss: binary_crossentropy")
print("  Metrics: accuracy, precision, recall, auc, tp, fp, tn, fn")

# Model summary
print(f"\nTotal parameters: {model.count_params():,}")
trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"Trainable parameters: {trainable_params:,}")

# Callbacks
print("\n" + "=" * 70)
print("CALLBACKS")
print("=" * 70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Model checkpoint - save best model based on validation accuracy
checkpoint_path = MODEL_DIR / f'skin_cancer_model_optimized_{timestamp}.keras'
checkpoint = ModelCheckpoint(
    str(checkpoint_path),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Early stopping - stop if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# CSV Logger
csv_path = BASE_DIR / f'training_log_optimized_{timestamp}.csv'
csv_logger = CSVLogger(str(csv_path))

callbacks = [checkpoint, early_stop, reduce_lr, csv_logger]

print("Callbacks configured:")
print(f"  1. ModelCheckpoint: Save best model (val_accuracy)")
print(f"     → {checkpoint_path}")
print(f"  2. EarlyStopping: Patience=10 (val_loss)")
print(f"  3. ReduceLROnPlateau: Factor=0.5, Patience=5")
print(f"  4. CSVLogger: Log metrics")
print(f"     → {csv_path}")

# Train Model
print("\n" + "=" * 70)
print("TRAINING START")
print("=" * 70)
print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("This may take a while...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Evaluate on test set
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_results = model.evaluate(test_generator, verbose=1)
metrics_names = model.metrics_names

print("\nTest Results:")
for name, value in zip(metrics_names, test_results):
    if 'loss' in name:
        print(f"  {name}: {value:.4f}")
    else:
        print(f"  {name}: {value:.4f} ({value*100:.2f}%)")

# Calculate additional metrics
# Safely get confusion matrix values
tp = test_results[metrics_names.index('tp')] if 'tp' in metrics_names else 0
fp = test_results[metrics_names.index('fp')] if 'fp' in metrics_names else 0
tn = test_results[metrics_names.index('tn')] if 'tn' in metrics_names else 0
fn = test_results[metrics_names.index('fn')] if 'fn' in metrics_names else 0

if tp > 0 or fp > 0 or tn > 0 or fn > 0:
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (Malignant correctly detected): {int(tp)}")
    print(f"  False Positives (Benign misclassified): {int(fp)}")
    print(f"  True Negatives (Benign correctly detected): {int(tn)}")
    print(f"  False Negatives (Malignant missed): {int(fn)}")
    print(f"\nAdditional Metrics:")
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  F1 Score: {f1_score:.4f} ({f1_score*100:.2f}%)")

# Save final model
final_model_path = MODEL_DIR / 'skin_cancer_model_optimized_final.keras'
model.save(str(final_model_path))
print(f"\n✓ Final model saved: {final_model_path}")

# Save class names
class_names_path = MODEL_DIR / 'class_names.txt'
with open(class_names_path, 'w') as f:
    for class_name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
        f.write(f"{class_name}\n")
print(f"✓ Class names saved: {class_names_path}")

# Plot training history
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Training History - Optimized Model', fontsize=16, fontweight='bold')

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Accuracy', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Loss', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[0, 2].plot(history.history['precision'], label='Train', linewidth=2)
axes[0, 2].plot(history.history['val_precision'], label='Validation', linewidth=2)
axes[0, 2].set_title('Precision', fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Precision')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Recall
axes[1, 0].plot(history.history['recall'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_recall'], label='Validation', linewidth=2)
axes[1, 0].set_title('Recall', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# AUC
axes[1, 1].plot(history.history['auc'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_auc'], label='Validation', linewidth=2)
axes[1, 1].set_title('AUC', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Learning Rate
if 'lr' in history.history:
    axes[1, 2].plot(history.history['lr'], linewidth=2, color='red')
    axes[1, 2].set_title('Learning Rate', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
else:
    axes[1, 2].axis('off')

plt.tight_layout()
plot_path = BASE_DIR / f'training_history_optimized_{timestamp}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Training plots saved: {plot_path}")

# Summary
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

best_epoch = np.argmax(history.history['val_accuracy'])
print(f"\nBest Model (Epoch {best_epoch + 1}):")
print(f"  Training Accuracy: {history.history['accuracy'][best_epoch]:.4f} ({history.history['accuracy'][best_epoch]*100:.2f}%)")
print(f"  Validation Accuracy: {history.history['val_accuracy'][best_epoch]:.4f} ({history.history['val_accuracy'][best_epoch]*100:.2f}%)")
print(f"  Validation Precision: {history.history['val_precision'][best_epoch]:.4f} ({history.history['val_precision'][best_epoch]*100:.2f}%)")
print(f"  Validation Recall: {history.history['val_recall'][best_epoch]:.4f} ({history.history['val_recall'][best_epoch]*100:.2f}%)")
print(f"  Validation AUC: {history.history['val_auc'][best_epoch]:.4f}")

print(f"\nFinal Test Performance:")
# Safely access metrics
for i, name in enumerate(metrics_names):
    if name in ['accuracy', 'precision', 'recall', 'auc']:
        print(f"  {name.capitalize()}: {test_results[i]*100:.2f}%")
    elif name not in ['loss', 'tp', 'fp', 'tn', 'fn']:
        print(f"  {name}: {test_results[i]:.4f}")

if tp > 0 or fp > 0 or tn > 0 or fn > 0:
    print(f"  F1 Score: {f1_score*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")

print("\n" + "=" * 70)
print("DONE! Model is ready to use.")
print("=" * 70)
print(f"\nModel saved at: {final_model_path}")
print(f"Training log: {csv_path}")
print(f"Training plots: {plot_path}")
print("\nTo use the model in the app, update the model path in app.py")
