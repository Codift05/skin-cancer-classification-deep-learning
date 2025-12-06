#!/usr/bin/env python3
"""
Evaluate model with different thresholds to find optimal balance
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

print("="*80)
print("THRESHOLD OPTIMIZATION FOR SKIN CANCER MODEL")
print("="*80)

# Load model
model_path = "model/skin_cancer_model.keras"
print(f"\n📂 Loading model: {model_path}")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully")

# Load test data
test_dir = "data/test"
img_size = (224, 224)
batch_size = 32

print(f"\n📂 Loading test dataset from: {test_dir}")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary',
    class_names=['benign', 'malignant']
)

# Normalize
normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
test_ds_normalized = test_ds.map(lambda x, y: (normalization_layer(x), y))

print("✅ Test dataset loaded")

# Get predictions
print("\n🔮 Getting predictions on test set...")
predictions = []
true_labels = []

for images, labels in test_ds_normalized:
    preds = model.predict(images, verbose=0)
    predictions.extend(preds.flatten())
    true_labels.extend(labels.numpy())

predictions = np.array(predictions)
true_labels = np.array(true_labels).astype(int)

print(f"✅ Got {len(predictions)} predictions")
print(f"   Unique labels: {np.unique(true_labels)}")
print(f"   Label distribution: Benign={np.sum(true_labels==0)}, Malignant={np.sum(true_labels==1)}")

print(f"✅ Got {len(predictions)} predictions")

# Evaluate with different thresholds
print("\n" + "="*80)
print("THRESHOLD EVALUATION")
print("="*80)

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

results = []
for threshold in thresholds:
    # Apply threshold
    predicted_classes = (predictions > threshold).astype(int)
    
    # Calculate metrics
    tp = np.sum((predicted_classes == 1) & (true_labels == 1))  # True Positive
    tn = np.sum((predicted_classes == 0) & (true_labels == 0))  # True Negative
    fp = np.sum((predicted_classes == 1) & (true_labels == 0))  # False Positive
    fn = np.sum((predicted_classes == 0) & (true_labels == 1))  # False Negative
    
    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    })
    
    print(f"\n🎯 Threshold: {threshold:.2f}")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}% (dari {tp+fp} prediksi malignant, {tp} benar)")
    print(f"   Recall:    {recall*100:.2f}% (dari {tp+fn} malignant, {tp} terdeteksi)")
    print(f"   F1-Score:  {f1_score:.4f}")
    print(f"   TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# Find best threshold
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Best F1-score
best_f1 = max(results, key=lambda x: x['f1_score'])
print(f"\n🏆 Best F1-Score: {best_f1['threshold']:.2f}")
print(f"   Accuracy: {best_f1['accuracy']*100:.2f}%")
print(f"   Precision: {best_f1['precision']*100:.2f}%")
print(f"   Recall: {best_f1['recall']*100:.2f}%")
print(f"   F1: {best_f1['f1_score']:.4f}")

# Best accuracy
best_acc = max(results, key=lambda x: x['accuracy'])
print(f"\n📊 Best Accuracy: {best_acc['threshold']:.2f}")
print(f"   Accuracy: {best_acc['accuracy']*100:.2f}%")
print(f"   Precision: {best_acc['precision']*100:.2f}%")
print(f"   Recall: {best_acc['recall']*100:.2f}%")

# Best recall (important for medical diagnosis - don't miss cancer!)
best_recall = max(results, key=lambda x: x['recall'])
print(f"\n⚕️ Best Recall (Medical Safety): {best_recall['threshold']:.2f}")
print(f"   Accuracy: {best_recall['accuracy']*100:.2f}%")
print(f"   Precision: {best_recall['precision']*100:.2f}%")
print(f"   Recall: {best_recall['recall']*100:.2f}% ← Minimal false negatives!")
print(f"   Missed malignant: {best_recall['fn']} dari {best_recall['tp']+best_recall['fn']}")

print("\n" + "="*80)
print("💡 RECOMMENDATION:")
print(f"   Use threshold = {best_f1['threshold']:.2f} for balanced performance")
print(f"   Or threshold = {best_recall['threshold']:.2f} untuk medical safety (minimize missed cancer)")
print("="*80)
