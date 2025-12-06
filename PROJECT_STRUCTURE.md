# 📁 PROJECT STRUCTURE - Skin Cancer Classification

Complete project structure dan file overview.

```
skin_cancer_classification/
│
├── 📂 data/                    # Dataset folder (user provides)
│   ├── benign/                 # Benign (non-cancer) images
│   └── malignant/              # Malignant (cancer) images
│
├── 📂 notebook/                # Training & experimentation
│   └── training.ipynb          # ⭐ Main training notebook
│
├── 📂 model/                   # Trained model & outputs
│   ├── model.h5                # Saved MobileNetV2 model (generated after training)
│   ├── labels.txt              # Class labels file (generated after training)
│   └── gradcam_example.png     # Example Grad-CAM visualization (generated)
│
├── 📂 app/                     # Web application
│   ├── __init__.py             # Package initialization
│   └── app.py                  # ⭐ Streamlit web app
│
├── 📂 utils/                   # Utility modules
│   ├── __init__.py             # Package initialization
│   ├── preprocess.py           # Image preprocessing functions
│   ├── gradcam.py              # Grad-CAM visualization module
│   └── helpers.py              # Helper utilities
│
├── 📂 .streamlit/              # Streamlit configuration
│   └── config.toml             # Theme & settings
│
├── 📂 test/                    # Test images (user provided)
│   ├── benign/
│   └── malignant/
│
├── 📂 train/                   # Training images (user provided)
│   ├── benign/
│   └── malignant/
│
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # 📖 Main documentation
├── 📄 SETUP_GUIDE.md            # 🚀 Installation guide
├── 📄 API_REFERENCE.md          # 📚 API documentation
├── 📄 .gitignore               # Git ignore rules
├── 📄 run.bat                  # Windows batch script
└── 📄 run.sh                   # Linux/Mac shell script
```

## 📋 FILE OVERVIEW

### Core Files

| File | Purpose | Status |
|------|---------|--------|
| `notebook/training.ipynb` | Complete training pipeline | ✅ Ready |
| `app/app.py` | Streamlit web interface | ✅ Ready |
| `requirements.txt` | Python dependencies | ✅ Ready |

### Utility Modules

| Module | Functions | Status |
|--------|-----------|--------|
| `utils/preprocess.py` | Image loading & preparation | ✅ Ready |
| `utils/gradcam.py` | Grad-CAM visualization | ✅ Ready |
| `utils/helpers.py` | Helper utilities | ✅ Ready |

### Documentation

| Document | Content | Status |
|----------|---------|--------|
| `README.md` | Complete project documentation | ✅ Ready |
| `SETUP_GUIDE.md` | Installation & troubleshooting | ✅ Ready |
| `API_REFERENCE.md` | Detailed API documentation | ✅ Ready |

### Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `.streamlit/config.toml` | Streamlit theme settings | ✅ Ready |
| `.gitignore` | Git ignore rules | ✅ Ready |
| `run.bat` | Windows script | ✅ Ready |
| `run.sh` | Linux/Mac script | ✅ Ready |

## 🎯 QUICK START

### 1️⃣ Installation (5 minutes)
```bash
# Option A: Windows
run.bat
# Select option 4 (Install & setup)

# Option B: Linux/Mac
chmod +x run.sh
./run.sh
# Select option 4 (Install & setup)
```

### 2️⃣ Download Dataset
1. Visit: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign
2. Download dataset
3. Extract to `data/` folder

### 3️⃣ Train Model (30-60 minutes)
```bash
# Option A: Using script
run.bat              # Select option 2
# or
./run.sh             # Select option 2

# Option B: Manual
jupyter lab notebook/training.ipynb
# Run all cells in order
```

### 4️⃣ Run Web App
```bash
# Option A: Using script
run.bat              # Select option 3
# or
./run.sh             # Select option 3

# Option B: Manual
streamlit run app/app.py
```

### 5️⃣ Make Predictions
- Open browser: http://localhost:8501
- Upload image
- View results & Grad-CAM

## 📊 DATA STRUCTURE

### Dataset Organization
```
data/
├── benign/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ... (300+ images)
│
└── malignant/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ... (300+ images)
```

### Expected Dataset Stats
- **Total images:** 600-1000+
- **Benign:** ~50%
- **Malignant:** ~50%
- **Image format:** JPG/PNG
- **Resolution:** Variable (resized to 224x224)

## 🔧 DEVELOPMENT WORKFLOW

### Modify Training Parameters
**File:** `notebook/training.ipynb`
```python
# Cell: Train the Model
EPOCHS = 25              # Change epochs
BATCH_SIZE = 32          # Change batch size
IMG_SIZE = (224, 224)    # Change input size
```

### Modify Web App UI
**File:** `app/app.py`
```python
# Customize colors
class_colors = {
    'benign': '#00AA00',     # Green
    'malignant': '#FF0000',  # Red
}

# Customize messages
# Look for st.markdown() and st.write() calls
```

### Add New Preprocessing Functions
**File:** `utils/preprocess.py`
```python
def your_function(img_array):
    # Add your preprocessing logic
    return processed_img
```

## 📦 PROJECT COMPONENTS

### 1. Training Component
- **Input:** Images in `data/` folder
- **Process:** MobileNetV2 + transfer learning
- **Output:** `model.h5`, `labels.txt`, `gradcam_example.png`

### 2. Inference Component
- **Input:** Trained model + new image
- **Process:** Preprocessing → Prediction
- **Output:** Class prediction + confidence

### 3. Visualization Component
- **Input:** Trained model + image
- **Process:** Grad-CAM heatmap generation
- **Output:** Heatmap + overlay image

### 4. Web Component
- **Framework:** Streamlit
- **Features:** Upload, preview, predict, visualize
- **Output:** Interactive web interface

## 🚀 DEPLOYMENT

### Local Development
```bash
streamlit run app/app.py
```

### Docker Deployment (Optional)
```bash
# Build Dockerfile (if provided)
docker build -t skin-cancer-app .
docker run -p 8501:8501 skin-cancer-app
```

### Cloud Deployment Options
- **Streamlit Cloud:** Free hosting
- **Heroku:** Traditional hosting
- **AWS/Azure/GCP:** Scalable deployment

## 🧪 TESTING

### Test Installation
```bash
python test_setup.py
```

### Test Model
```python
from utils.preprocess import load_and_prepare
from tensorflow.keras.models import load_model

model = load_model('model/model.h5')
img = load_and_prepare('path/to/test/image.jpg')
prediction = model.predict(img[np.newaxis, ...])
print(f"Prediction: {prediction[0][0]:.4f}")
```

## 📈 PROJECT STATISTICS

### Code Statistics
- **Total Python files:** 6 (preprocess.py, gradcam.py, helpers.py, app.py, __init__.py files)
- **Total lines of code:** ~2000+
- **Notebook cells:** 20+

### Model Statistics
- **Base model parameters:** 2.2M (MobileNetV2)
- **Custom head parameters:** 130K
- **Total trainable parameters:** 130K
- **Total model size:** ~15-20 MB

### Dataset Statistics
- **Training images:** ~80% (640-800)
- **Validation images:** ~20% (160-200)
- **Image dimensions:** 224x224x3
- **Data augmentation:** 4 types

## 🔗 DEPENDENCIES

### Core Libraries
- TensorFlow 2.15.0
- NumPy 1.24.3
- OpenCV 4.8.1
- Streamlit 1.28.1
- scikit-learn 1.3.1
- Matplotlib 3.7.2
- Pillow 10.0.0

### Version Compatibility
- Python: 3.8-3.11
- GPU: CUDA 12.0+ (optional)
- OS: Windows, macOS, Linux

## 💾 FILE SIZES

Expected file sizes after training:

| File | Size |
|------|------|
| model.h5 | ~18 MB |
| requirements packages | ~500 MB |
| gradcam_example.png | ~200 KB |
| Total project | ~550 MB |

## ⚙️ CONFIGURATION FILES

### .streamlit/config.toml
Streamlit theme & configuration:
- Primary color: #1f77b4 (Blue)
- Background: White
- Font: Sans serif

### requirements.txt
Contains all Python dependencies with pinned versions.

### .gitignore
Excludes large files and temporary data:
- Model files (.h5, .pkl)
- Cache files (__pycache__)
- Virtual environment (venv/)
- IDE files (.vscode, .idea)

## 🎓 LEARNING RESOURCES

### Inside Project
1. **training.ipynb** - Learn ML pipeline step-by-step
2. **API_REFERENCE.md** - Understand utility functions
3. **SETUP_GUIDE.md** - Detailed setup instructions

### External Resources
- TensorFlow: https://www.tensorflow.org
- Streamlit: https://docs.streamlit.io
- MobileNetV2: https://arxiv.org/abs/1801.04381
- Grad-CAM: https://arxiv.org/abs/1610.02055

---

**Project Status:** ✅ Complete & Ready to Use  
**Last Updated:** December 2024  
**Version:** 1.0.0
