# 🚀 Setup Guide - Skin Cancer Classification Project

Complete guide untuk setup dan menjalankan project.

## 📋 Persyaratan Sistem

- **OS:** Windows 10+, macOS 10.14+, atau Linux (Ubuntu 18.04+)
- **Python:** 3.8 - 3.11 (recommended: 3.9 or 3.10)
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** 5GB (untuk model + dependencies)
- **GPU:** Optional tapi recommended (NVIDIA dengan CUDA)

## 📥 Step-by-Step Installation

### Step 1: Download Dataset

1. Kunjungi [Kaggle - Skin Cancer Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)
2. Download dataset
3. Extract ke folder `data/` dengan struktur:
   ```
   data/
   ├── benign/
   └── malignant/
   ```

### Step 2: Setup Environment

#### Option A: Automatic Setup (Recommended)

**Windows:**
```bash
run.bat
# Pilih opsi 4 (Install & setup)
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
# Pilih opsi 4 (Install & setup)
```

#### Option B: Manual Setup

**1. Buat Virtual Environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Verify Installation**

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import streamlit as st; print('Streamlit OK')"
```

### Step 3: Verify Dataset

```bash
# Check folder structure
# Windows
dir data

# Linux/Mac
ls -la data/
```

Output yang diharapkan:
```
data/
├── benign/         (300+ images)
└── malignant/      (300+ images)
```

## 🎓 Menjalankan Project

### 1. Training Model (First Time Only)

**Using Script:**
```bash
# Windows
run.bat          # Pilih opsi 2

# Linux/Mac
./run.sh         # Pilih opsi 2
```

**Manual:**
```bash
jupyter lab notebook/training.ipynb
```

**Langkah dalam notebook:**
1. Run cell pertama untuk import libraries
2. Run cell untuk load dataset
3. Run cell untuk preprocessing & augmentation
4. Run cell untuk build & train model
5. Run cell untuk evaluation
6. Run cell untuk save model & generate Grad-CAM

**Durasi:** 30-60 menit (tergantung CPU/GPU)

**Output yang dihasilkan:**
- `model/model.h5` - Trained model
- `model/labels.txt` - Class labels
- `model/gradcam_example.png` - Grad-CAM visualization

### 2. Jalankan Web Application

**Using Script:**
```bash
# Windows
run.bat          # Pilih opsi 3

# Linux/Mac
./run.sh         # Pilih opsi 3
```

**Manual:**
```bash
streamlit run app/app.py
```

**Access aplikasi:**
- Browser akan otomatis membuka: http://localhost:8501
- Jika tidak, buka manual di browser

## 🛠️ Troubleshooting

### Issue 1: Python version error
```bash
# Check Python version
python --version

# Harus 3.8 atau lebih baru
# Jika tidak, install dari python.org
```

### Issue 2: pip install error
```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall requirements
pip install --no-cache-dir -r requirements.txt
```

### Issue 3: CUDA/GPU not found
```bash
# Jika pakai GPU NVIDIA:
pip install tensorflow[and-cuda]==2.15.0

# Atau gunakan CPU (slower):
# Aplikasi akan otomatis fallback ke CPU
```

### Issue 4: Out of Memory
**Solusi:**
- Kurangi `BATCH_SIZE` di notebook (32 → 16)
- Kurangi `EPOCHS` (25 → 15)
- Gunakan GPU jika tersedia

### Issue 5: Streamlit tidak start
```bash
# Clear Streamlit cache
streamlit cache clear

# Jalankan dengan verbose logging
streamlit run app/app.py --logger.level=debug
```

### Issue 6: Model file not found
```bash
# Pastikan folder structure:
# model/
# ├── model.h5
# ├── labels.txt
# └── gradcam_example.png

# Jika tidak ada, jalankan training notebook terlebih dahulu
```

## 🧪 Test Installation

Jalankan test script berikut untuk verify setup:

```python
# test_setup.py
import sys
from pathlib import Path

def test_imports():
    """Test semua library dapat diimport"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        import streamlit as st
        print("✅ Streamlit")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import cv2
        print("✅ OpenCV")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
        
        from sklearn.metrics import accuracy_score
        print("✅ scikit-learn")
        
        return True
    except ImportError as e:
        print(f"❌ Error: {e}")
        return False

def test_paths():
    """Test folder structure"""
    base = Path(__file__).parent
    
    required_dirs = [
        'data/benign',
        'data/malignant',
        'notebook',
        'model',
        'app',
        'utils',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (not found)")
            all_exist = False
    
    return all_exist

def test_dataset():
    """Test dataset availability"""
    base = Path(__file__).parent
    
    benign_count = len(list((base / 'data' / 'benign').glob('*.*')))
    malignant_count = len(list((base / 'data' / 'malignant').glob('*.*')))
    
    print(f"\nDataset:")
    print(f"  Benign: {benign_count} images")
    print(f"  Malignant: {malignant_count} images")
    print(f"  Total: {benign_count + malignant_count} images")
    
    return benign_count > 0 and malignant_count > 0

if __name__ == "__main__":
    print("Testing installation...\n")
    
    print("1. Testing imports...")
    imports_ok = test_imports()
    
    print("\n2. Testing folder structure...")
    paths_ok = test_paths()
    
    print("\n3. Testing dataset...")
    dataset_ok = test_dataset()
    
    print("\n" + "="*40)
    if imports_ok and paths_ok and dataset_ok:
        print("✅ All tests passed! Ready to go!")
    elif imports_ok and paths_ok:
        print("⚠️  Dependencies OK, but dataset missing")
        print("   Download from: https://www.kaggle.com/.../skin-cancer-malignant-vs-benign")
    else:
        print("❌ Some issues found. See above.")
    print("="*40)
```

Jalankan dengan:
```bash
python test_setup.py
```

## 📚 Workflow Summary

```
1. Setup (install dependencies)
   ↓
2. Dataset (download & organize)
   ↓
3. Training (run notebook, ~45 min)
   ↓
4. Model saved (model.h5 created)
   ↓
5. Web App (streamlit run)
   ↓
6. Make predictions!
```

## 🎯 Next Steps

Setelah setup selesai:

1. **Train Model** → Run `notebook/training.ipynb`
2. **Test Web App** → Run `streamlit run app/app.py`
3. **Make Predictions** → Upload image & analyze
4. **Understand Results** → Check Grad-CAM visualization

## 📞 Getting Help

Jika mengalami masalah:

1. **Check README.md** - Dokumentasi lengkap
2. **Check errors output** - Error message sering sangat informatif
3. **Check dependencies** - Pastikan semua library terinstall
4. **Check dataset** - Pastikan file ada di tempat yang benar

## ✅ Success Indicators

Anda siap jika:

- ✅ Semua dependencies terinstall tanpa error
- ✅ Folder structure lengkap
- ✅ Dataset ada di `data/benign` dan `data/malignant`
- ✅ Jupyter Lab dapat membuka notebook
- ✅ Streamlit dapat dijalankan
- ✅ Model training berjalan tanpa error

---

**Pertanyaan umum?** Lihat README.md untuk FAQ lengkap.
