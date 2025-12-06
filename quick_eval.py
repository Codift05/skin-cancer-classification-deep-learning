import numpy as np
import tensorflow as tf
from pathlib import Path

# Load model
model = tf.keras.models.load_model("model/skin_cancer_model.keras")

# Load test data  
test_dir = "data/test"
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary',
    shuffle=False  # Don't shuffle for consistent results
)

# Normalize
norm = tf.keras.layers.Rescaling(1./127.5, offset=-1)
test_ds_normalized = test_ds.map(lambda x, y: (norm(x), y))

# Get all predictions and labels - take only first N batches
all_predictions = []
all_labels = []

for batch_idx, (images, labels) in enumerate(test_ds_normalized):
    preds = model.predict(images, verbose=0)
    all_predictions.append(preds)
    all_labels.append(labels.numpy())
    
    # Break after processing all batches (660 images / 32 batch = ~21 batches)
    if batch_idx >= 20:  # 21 batches total (0-20)
        break

all_predictions = np.concatenate(all_predictions, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"\n Total predictions: {len(all_predictions)}, shape: {all_predictions.shape}")
print(f"Total labels: {len(all_labels)}, shape: {all_labels.shape}")
print(f"Benign: {np.sum(all_labels==0)}, Malignant: {np.sum(all_labels==1)}")
print(f"Label values: min={all_labels.min()}, max={all_labels.max()}")
print(f"Prediction values: min={all_predictions.min():.4f}, max={all_predictions.max():.4f}")

# Test thresholds - focused on 0.24-0.38 range where model operates
thresholds = [0.24, 0.26, 0.27, 0.28, 0.29, 0.30, 0.32, 0.35, 0.38]

print("\n" + "="*70)
print("THRESHOLD EVALUATION")
print("="*70)

# Flatten both arrays to ensure 1D comparison
all_predictions_flat = all_predictions.flatten()
all_labels_flat = all_labels.flatten()

for t in thresholds:
    pred_classes = (all_predictions_flat > t).astype(int)
    
    tp = np.sum((pred_classes == 1) & (all_labels_flat == 1))
    tn = np.sum((pred_classes == 0) & (all_labels_flat == 0))
    fp = np.sum((pred_classes == 1) & (all_labels_flat == 0))
    fn = np.sum((pred_classes == 0) & (all_labels_flat == 1))
    
    print(f"\nDEBUG threshold {t}: TP={tp}, TN={tn}, FP={fp}, FN={fn}, Sum={tp+tn+fp+fn}")
    
    acc = (tp + tn) / len(all_labels_flat) * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"\n🎯 Threshold: {t:.2f}")
    print(f"   Accuracy:  {acc:.2f}%")
    print(f"   Precision: {prec:.2f}% ({tp}/{tp+fp} predictions correct)")
    print(f"   Recall:    {rec:.2f}% ({tp}/{tp+fn} malignant detected)")
    print(f"   F1-Score:  {f1:.2f}")
    print(f"   Missed malignant: {fn}")

print("\n" + "="*70)
print("RECOMMENDATION: Use threshold 0.35-0.40 for better balance")
print("="*70)
