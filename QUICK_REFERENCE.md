# ✅ QUICK REFERENCE & CHECKLIST

## 📝 Pre-Setup Checklist

- [ ] Python 3.8+ installed (`python --version`)
- [ ] pip installed (`pip --version`)
- [ ] Kaggle dataset downloaded
- [ ] Dataset extracted to `data/` folder
- [ ] 5GB free disk space available
- [ ] 4GB+ RAM available

## 🚀 Setup Checklist

- [ ] Step 1: Install dependencies (`pip install -r requirements.txt`)
- [ ] Step 2: Dataset folder structure verified
- [ ] Step 3: Virtual environment created (recommended)
- [ ] Step 4: Test imports working

## 📚 Training Checklist

- [ ] Dataset images count verified (600+ total)
- [ ] Open `notebook/training.ipynb`
- [ ] Run cell 1: Import libraries
- [ ] Run cell 2: Load dataset
- [ ] Run cell 3: Explore data
- [ ] Run cell 4: Preprocessing & augmentation
- [ ] Run cell 5: Build model (MobileNetV2)
- [ ] Run cell 6: Compile model
- [ ] Run cell 7: Train model (30-60 min)
- [ ] Run cell 8: Evaluate model
- [ ] Run cell 9: Save model & labels
- [ ] Run cell 10: Generate Grad-CAM
- [ ] Verify `model/model.h5` created (18 MB)
- [ ] Verify `model/labels.txt` created
- [ ] Verify `model/gradcam_example.png` created

## 🌐 Web App Checklist

- [ ] Model file exists: `model/model.h5`
- [ ] Labels file exists: `model/labels.txt`
- [ ] Run command: `streamlit run app/app.py`
- [ ] Browser opens to http://localhost:8501
- [ ] Upload sample image
- [ ] Prediction works
- [ ] Grad-CAM visualization displays

## 📊 Common Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install specific package
pip install tensorflow==2.15.0

# List installed packages
pip list
```

### Training
```bash
# Open notebook
jupyter lab notebook/training.ipynb

# Or with Jupyter
jupyter notebook notebook/training.ipynb
```

### Web Application
```bash
# Run Streamlit app
streamlit run app/app.py

# Run with custom port
streamlit run app/app.py --server.port 8502

# Clear cache
streamlit cache clear
```

### Utilities
```bash
# Test installation
python test_setup.py

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check dataset
python -c "from pathlib import Path; b = len(list(Path('data/benign').glob('*'))); m = len(list(Path('data/malignant').glob('*'))); print(f'Benign: {b}, Malignant: {m}, Total: {b+m}')"
```

## 🔍 Quick Troubleshooting

### "Module not found"
```bash
pip install --upgrade -r requirements.txt
```

### "GPU not detected"
- Install NVIDIA drivers
- Or use CPU (slower but works)

### "Out of Memory"
- Close other applications
- Reduce BATCH_SIZE in notebook
- Reduce EPOCHS

### "Port already in use"
```bash
streamlit run app/app.py --server.port 8502
```

## 📂 Important Files

| File | Purpose | When Needed |
|------|---------|------------|
| `notebook/training.ipynb` | Train model | First time |
| `model/model.h5` | Trained model | Always (generated from training) |
| `model/labels.txt` | Class labels | Always (generated from training) |
| `app/app.py` | Web interface | Always |
| `utils/preprocess.py` | Image preprocessing | Used by app & notebook |
| `utils/gradcam.py` | Grad-CAM visualization | Used by app & notebook |

## 🎯 Project Workflow

```
1. SETUP (5 min)
   └─ pip install -r requirements.txt
   
2. DATASET (5 min)
   └─ Download & organize data/
   
3. TRAINING (45 min)
   └─ jupyter lab notebook/training.ipynb
   └─ Generate: model.h5, labels.txt, gradcam_example.png
   
4. WEB APP (1 min setup)
   └─ streamlit run app/app.py
   └─ Upload image → Get prediction → View Grad-CAM
   
5. DEPLOYMENT (optional)
   └─ Deploy to cloud (Streamlit Cloud, Heroku, etc.)
```

## 🎓 Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Image size | 224x224 | training.ipynb |
| Batch size | 32 | training.ipynb |
| Learning rate | 1e-4 | training.ipynb |
| Epochs | 25 | training.ipynb |
| Dropout | 0.3 | training.ipynb |
| Validation split | 0.2 | training.ipynb |

## 📊 Expected Metrics

After training:
- **Accuracy:** 85-95%
- **Precision:** 85-92%
- **Recall:** 88-95%
- **F1-Score:** 0.87-0.93

*Exact metrics depend on dataset quality*

## 🔗 Useful Links

### Documentation
- README.md - Main documentation
- SETUP_GUIDE.md - Installation guide
- API_REFERENCE.md - API docs
- PROJECT_STRUCTURE.md - Project overview

### External Resources
- TensorFlow: https://www.tensorflow.org
- Streamlit: https://docs.streamlit.io
- Kaggle Dataset: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign
- MobileNetV2 Paper: https://arxiv.org/abs/1801.04381

## 🆘 Emergency Fixes

### Reset Everything
```bash
# Remove model files
rm model/model.h5 model/labels.txt model/gradcam_example.png

# Clear Python cache
rm -rf __pycache__ .ipynb_checkpoints

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Clear Streamlit Cache
```bash
streamlit cache clear
rm -rf ~/.streamlit
```

### Reinstall Virtual Environment
```bash
# Windows
rmdir venv /s
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📞 Support Resources

1. **Check error message** - Often very descriptive
2. **Google the error** - Usually common issues
3. **Check GitHub issues** - Similar problems solved
4. **Consult documentation** - README.md, SETUP_GUIDE.md
5. **Try minimal example** - Test with single image

## ✨ Next Steps After Setup

1. ✅ Complete setup
2. ✅ Train model (wait for completion)
3. ✅ Test web app with sample image
4. ✅ Explore Grad-CAM visualizations
5. ✅ Understand model predictions
6. ✅ Deploy (optional)
7. ✅ Share with others

## 📈 Performance Tips

### Faster Training
- Use GPU (CUDA)
- Reduce image resolution
- Reduce batch size
- Use smaller model (MobileNetV2 is already small)

### Better Accuracy
- More training data
- Longer training time
- Better data augmentation
- Hyperparameter tuning

### Faster Inference
- Use TensorFlow Lite
- Quantization
- Model pruning
- Deploy to edge device

## 🎯 Success Criteria

✅ Project is successful when:
1. Model trains without errors
2. Model saves to `model/model.h5`
3. Web app starts without errors
4. Web app makes predictions
5. Grad-CAM visualization displays
6. Predictions are reasonable (80%+ accuracy)

## 🚀 Ready to Start?

1. **Read:** README.md (5 min)
2. **Setup:** SETUP_GUIDE.md (10 min)
3. **Install:** requirements.txt (5 min)
4. **Download:** Dataset (depends on connection)
5. **Train:** notebook/training.ipynb (45 min)
6. **Run:** app/app.py (1 min)

**Total time:** ~1 hour (mostly training)

---

**Questions?** Check the documentation files!  
**Stuck?** Try the troubleshooting section above.  
**Ready?** Let's get started! 🎉
