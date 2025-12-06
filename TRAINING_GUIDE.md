# 🎯 Training Guide - Skin Cancer Classification

## Current Status
- ✅ Dataset loaded: **2,637 images** (1,440 benign + 1,197 malignant)
- ⏳ TensorFlow: Installing...
- ⏳ Environment: Setting up

## Option 1: Using Jupyter Notebook (Recommended)

### Steps:
1. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

2. **Open notebook:**
   - Navigate to: `notebook/training.ipynb`
   - Click on first cell and press `Shift+Enter` to start
   - Press `Shift+Enter` on each cell sequentially

3. **What the notebook does:**
   - Cell 1-3: Import libraries & check dataset
   - Cell 4-7: Load and visualize data
   - Cell 8-10: Preprocessing & augmentation
   - Cell 11-13: Build MobileNetV2 model
   - Cell 14-16: Train the model (this takes ~10-30 minutes)
   - Cell 17-19: Evaluate results
   - Cell 20+: Save model and Grad-CAM examples

### Expected output:
```
Epoch 1/25
120/120 ━━━━━━━━━━━━━━━━━━━━━━━ 45s 375ms/step - loss: 0.8234 - accuracy: 0.6521 - val_loss: 0.5123 - val_accuracy: 0.7234
...
Epoch 25/25
120/120 ━━━━━━━━━━━━━━━━━━━━━━━━ 32s 267ms/step - loss: 0.1234 - accuracy: 0.9456 - val_loss: 0.2567 - val_accuracy: 0.8901
```

---

## Option 2: Using Training Script

### Steps:
1. **Check setup:**
   ```bash
   python check_setup.py
   ```

2. **Run training:**
   ```bash
   python train_simple.py
   ```

3. **Monitor progress:**
   ```
   [9/10] Training model...
   Epoch 1/25...
   ```

---

## Option 3: Manual Training (Advanced)

```python
import tensorflow as tf
from pathlib import Path

# Load data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary',
    class_names=['benign', 'malignant']
)

# Load MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(train_ds, epochs=25)

# Save
model.save('model/skin_cancer_model.h5')
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:**
```bash
pip install tensorflow
```
*First installation takes ~10-15 minutes*

### ❌ "No images found in data/train"
**Solution:** Make sure dataset is in correct structure:
```
data/train/
  ├── benign/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── malignant/
      ├── image1.jpg
      ├── image2.jpg
      └── ...
```

### ❌ "CUDA not available" (GPU warning)
**This is OK!** Training will use CPU. GPU training is optional and faster.
```
I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions
```

### ⏳ Training is very slow
**Possible causes:**
1. Using CPU instead of GPU (expected, ~20-40 min per epoch)
2. Dataset loading from slow disk (try copying to SSD)
3. Other programs consuming resources (close them)

**To speed up:**
- Reduce batch size: Change `batch_size=32` to `batch_size=16`
- Reduce epochs: Change `epochs=25` to `epochs=10`
- Use GPU: Install tensorflow-gpu and CUDA toolkit

---

## Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu)
    ↓
Dropout(0.3)
    ↓
Dense(1, sigmoid) → Output (benign=0, malignant=1)
```

**Parameters:** ~2.3M trainable + 2.2M frozen

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | MobileNetV2 |
| Input Size | 224x224 |
| Batch Size | 32 |
| Epochs | 25 |
| Optimizer | Adam (lr=1e-4) |
| Loss | Binary Crossentropy |
| Data Split | 80% train, 20% validation |
| Augmentation | Flip, Rotation, Zoom, Brightness |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

---

## Expected Training Time

| Device | Time per epoch | Total (25 epochs) |
|--------|----------------|-------------------|
| CPU (moderate) | 30-45s | 12-19 minutes |
| CPU (slow) | 60-90s | 25-38 minutes |
| GPU (RTX 3060) | 8-12s | 3-5 minutes |
| GPU (RTX 4090) | 3-5s | 1-2 minutes |

---

## After Training

### ✅ Generated Files:
- `model/skin_cancer_model.h5` - Trained model
- `model/class_names.txt` - Class labels

### 🚀 Next Steps:
1. **Run web app:**
   ```bash
   streamlit run app/app.py
   ```

2. **Upload test images and get predictions**

3. **View Grad-CAM visualizations**

---

## Success Indicators

✅ You'll know training is successful when:
- Model saves without errors
- Final val_accuracy > 75%
- model/skin_cancer_model.h5 exists (~240MB)
- Web app launches with `streamlit run app/app.py`

---

## Additional Resources

- 📖 **Notebook:** `notebook/training.ipynb` - Full walkthrough
- 📚 **API Reference:** `docs/API_REFERENCE.md` - Function details
- 🔧 **Quick Reference:** `docs/QUICK_REFERENCE.md` - Commands
- 📝 **README:** `README.md` - Project overview

---

*Last updated: 2025-12-06*
