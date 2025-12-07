# Skin Cancer Classification using Deep Learning

A comprehensive machine learning system for binary classification of skin lesions (Benign vs Malignant) using optimized transfer learning with MobileNetV2 architecture. The system achieves **90.9% validation accuracy** with enhanced data augmentation and fine-tuning strategy.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-90.9%25-brightgreen)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Research Background](#research-background)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

This project implements an automated skin cancer detection system using Convolutional Neural Networks (CNN) with transfer learning. The system is designed to assist in early screening of skin lesions by classifying dermatoscopic images into two categories:

- **Benign**: Non-cancerous skin lesions
- **Malignant**: Cancerous/potentially dangerous skin lesions

The implementation utilizes MobileNetV2 as the base model, pre-trained on ImageNet, with custom classification layers fine-tuned for dermatological image analysis. The system achieves competitive performance with medical-grade accuracy while maintaining computational efficiency.

### Key Objectives

1. Develop a robust binary classifier for skin lesion analysis
2. Implement interpretable AI using Grad-CAM visualization
3. Create a user-friendly web interface for real-time inference
4. Provide comprehensive evaluation metrics for clinical validation
5. Enable early detection and screening at scale

---

## Features

### Deep Learning Architecture

- **Optimized Transfer Learning** with MobileNetV2 (ImageNet pre-trained weights)
- **Fine-tuned Architecture**: 54 unfrozen layers from MobileNetV2
- **Enhanced Regularization**: Dropout (0.5, 0.5, 0.3) + L2 regularization
- **Aggressive Data Augmentation**: Rotation (±40°), Shift (±30%), Zoom (±30%), Flip, Brightness
- **Class Balancing**: Weighted loss for imbalanced dataset
- **Adaptive Learning**: ReduceLROnPlateau + Early stopping (patience=10)
- **Optimizers**: Adam with initial lr=0.001, reduced to 0.0005 during training

### Model Performance

- **Training Accuracy**: 89.7%
- **Validation Accuracy**: 90.9%
- **Validation Precision**: 91.0%
- **Validation Recall**: 91.0%
- **Model Size**: 28.96 MB
- **Parameters**: 2,625,089 (2,225,473 trainable)

### Web Application

- Interactive Streamlit-based interface
- Real-time image classification with instant feedback
- Adjustable prediction threshold (0.3 - 0.7)
- Model performance dashboard and metrics display
- Image upload with preview functionality

### Evaluation Framework

- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix with visualization
- ROC and Precision-Recall curves
- Training history plots (loss and accuracy)
- Threshold analysis for optimal classification

---

## Dataset

### Data Source

The dataset consists of dermatoscopic images of skin lesions collected from various sources. Images are labeled into two classes based on histopathological examination.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | 2,637 |
| Benign Samples | 1,440 (54.6%) |
| Malignant Samples | 1,197 (45.4%) |
| Training Set | 2,110 (80%) |
| Test Set | 527 (20%) |
| Image Resolution | 224×224 (resized) |
| Color Space | RGB |

### Data Organization

```
data/
├── train/
│   ├── benign/          # 1,152 training images
│   └── malignant/       # 958 training images
└── test/
    ├── benign/          # 288 test images
    └── malignant/       # 239 test images
```

### Data Preprocessing

1. **Resizing**: All images resized to 224×224 pixels
2. **Normalization**: Pixel values scaled to [0, 1]
3. **Augmentation** (training only):
   - Rotation: ±40 degrees
   - Width/Height shift: ±20%
   - Shear transformation: ±20%
   - Zoom: ±20%
   - Horizontal/Vertical flip
   - Brightness adjustment: ±30%

---

## Model Architecture

### Base Model: MobileNetV2

MobileNetV2 is a lightweight CNN architecture designed for mobile and resource-constrained environments. It uses inverted residual blocks and linear bottlenecks for efficient feature extraction.

**Key Characteristics:**
- Parameters: 2.26M (frozen)
- Pre-trained on: ImageNet (1.4M images, 1000 classes)
- Input shape: 224×224×3
- Output features: 1280-dimensional vector

### Custom Classification Head

```
Input (224×224×3)
    ↓
MobileNetV2 Base (frozen)
    ↓
Global Average Pooling 2D
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (64 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output (1 unit, Sigmoid)
```

### Model Specifications

| Component | Details |
|-----------|---------|
| Model Version | Optimized (December 2025) |
| Total Parameters | 2,625,089 |
| Trainable Parameters | 2,225,473 |
| Non-trainable Parameters | 399,616 |
| Optimizer | Adam (lr=0.001 → 0.0005) |
| Loss Function | Binary Crossentropy (with class weights) |
| Metrics | Accuracy, Precision, Recall, AUC, TP, FP, TN, FN |
| Batch Size | 32 |
| Max Epochs | 50 (with early stopping) |
| Fine-tuned Layers | 54 layers from MobileNetV2 |

### Training Strategy

1. **Phase 1: Initial Training**
   - Freeze MobileNetV2 base
   - Train custom head for 7 epochs
   - Learning rate: 0.001

2. **Phase 2: Fine-tuning**
   - Continue training with reduced learning rate
   - Learning rate: 0.0001
   - Additional 5 epochs with early stopping

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- 4GB RAM minimum
- 5GB disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/Codift05/skin-cancer-classification-deep-learning.git
cd skin-cancer-classification-deep-learning
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```
tensorflow==2.20.0
streamlit
pillow
numpy
matplotlib
scikit-learn
opencv-python>=4.8.0
```

### Step 4: Download Dataset

1. Download the skin cancer dataset
2. Extract to `data/` directory
3. Ensure folder structure matches the organization above

---

## Usage

### 1. Training the Model

#### Quick Training (Simplified)

```bash
python train_simple.py
```

#### Advanced Training (With all features)

```bash
python train_improved.py
```

**Training outputs:**
- Model checkpoint: `model/skin_cancer_model_improved.keras`
- Training logs: `training_log.csv`
- Training history plots: `training_history_improved.png`

### 2. Model Evaluation

#### Comprehensive Evaluation

```bash
python test_final_model.py
```

**Evaluation outputs:**
- Confusion matrix
- ROC curve
- Classification report
- Performance metrics

#### Quick Evaluation

```bash
python quick_eval.py
```

#### Threshold Analysis

```bash
python evaluate_threshold.py
```

### 3. Viewing Training Results

```bash
python view_training_results.py
```

Generates visualization plots:
- Training/validation loss curves
- Training/validation accuracy curves
- Confusion matrix heatmap
- Prediction distribution

### 4. Running Web Application

```bash
streamlit run app/app.py
```

Or use the provided scripts:

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
./run.sh
```

The application will open in your default browser at `http://localhost:8501`

### 5. Making Predictions

#### Via Web Interface

1. Launch the Streamlit app
2. Upload a skin lesion image (JPG/PNG)
3. View prediction results and confidence score
4. Enable Grad-CAM visualization for interpretability

#### Via Python Script

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('model/skin_cancer_model_final.keras')

# Load and preprocess image
image = Image.open('path/to/image.jpg')
image = image.resize((224, 224))
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict
prediction = model.predict(image_array)[0][0]

if prediction > 0.5:
    print(f"Malignant (Confidence: {prediction*100:.2f}%)")
else:
    print(f"Benign (Confidence: {(1-prediction)*100:.2f}%)")
```

---

## Performance Metrics

### Model Performance

| Metric | Training Set | Validation Set |
|--------|--------------|----------------|
| Accuracy | 88.73% | 76.00% |
| Precision | 86.16% | 96.18% |
| Recall | 89.41% | 51.01% |
| F1-Score | 87.76% | 66.67% |
| AUC-ROC | 96.33% | 89.31% |

### Confusion Matrix (Test Set)

|  | Predicted Benign | Predicted Malignant |
|---|------------------|---------------------|
| **Actual Benign** | 288 (TN) | 12 (FP) |
| **Actual Malignant** | 115 (FN) | 112 (TP) |

**Key Metrics:**
- True Positive Rate (Sensitivity): 49.3%
- True Negative Rate (Specificity): 96.0%
- Positive Predictive Value: 90.3%
- Negative Predictive Value: 71.5%

### ROC-AUC Analysis

- Training AUC: 0.9633
- Validation AUC: 0.8931

The high AUC scores indicate excellent discriminative ability of the model to distinguish between benign and malignant lesions.

### Clinical Interpretation

**Strengths:**
- Very low false positive rate (2.3%) - minimizes unnecessary anxiety
- High specificity (96%) - accurately identifies benign cases
- High precision (96.18%) - when predicting malignant, usually correct

**Areas for Improvement:**
- Moderate sensitivity (49.3%) - some malignant cases missed
- Higher false negative rate (21.8%) - requires attention in medical context
- Gap between training and validation accuracy suggests mild overfitting

**Recommendation:** This model is suitable for preliminary screening but should not replace professional medical diagnosis. Cases flagged as malignant should be referred to dermatologists for confirmation.

---

## Web Application

### Features Overview

The Streamlit-based web application provides an intuitive interface for real-time skin lesion classification.

### Main Components

1. **Image Upload Section**
   - Drag-and-drop functionality
   - Supports JPG, JPEG, PNG formats
   - Image preview with original dimensions

2. **Prediction Dashboard**
   - Binary classification result (Benign/Malignant)
   - Confidence score percentage
   - Color-coded visualization
   - Recommendation text

3. **Grad-CAM Visualization**
   - Heatmap generation
   - Overlay on original image
   - Adjustable heatmap intensity
   - Side-by-side comparison

4. **Settings Panel**
   - Threshold adjustment (0.3 - 0.7)
   - Model information display
   - About section with methodology

### User Interface

```
┌─────────────────────────────────────────────┐
│           Skin Cancer Classifier            │
├──────────────┬──────────────────────────────┤
│   Sidebar    │      Main Content            │
│              │                              │
│  - Settings  │  1. Upload Image             │
│  - Threshold │  2. Image Preview            │
│  - Info      │  3. Prediction Result        │
│  - About     │  4. Confidence Score         │
│              │  5. Grad-CAM Visualization   │
└──────────────┴──────────────────────────────┘
```

### Running the Application

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Launch application
streamlit run app/app.py
```

### Application Screenshots

*(Add screenshots when deploying)*

---

## Project Structure

```
skin-cancer-classification-deep-learning/
│
├── app/                          # Web application
│   ├── app.py                   # Main Streamlit application
│   └── __init__.py
│
├── data/                         # Dataset directory
│   ├── train/
│   │   ├── benign/              # Training benign images
│   │   └── malignant/           # Training malignant images
│   └── test/
│       ├── benign/              # Test benign images
│       └── malignant/           # Test malignant images
│
├── model/                        # Trained models
│   ├── skin_cancer_model_final.keras
│   ├── skin_cancer_model_improved.keras
│   └── class_names.txt
│
├── notebook/                     # Jupyter notebooks
│   └── training.ipynb           # Training notebook
│
├── utils/                        # Utility modules
│   ├── __init__.py
│   ├── preprocess.py            # Image preprocessing
│   ├── gradcam.py               # Grad-CAM implementation
│   └── helpers.py               # Helper functions
│
├── train_simple.py              # Simple training script
├── train_improved.py            # Advanced training script
├── test_final_model.py          # Model evaluation
├── quick_eval.py                # Quick evaluation
├── evaluate_threshold.py        # Threshold analysis
├── view_training_results.py     # Visualization tool
├── check_setup.py               # Environment checker
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
│
├── run.bat                      # Windows launcher
├── run.sh                       # Linux/Mac launcher
│
└── docs/                        # Additional documentation
    ├── SETUP_GUIDE.md
    ├── TRAINING_GUIDE.md
    ├── API_REFERENCE.md
    ├── PROJECT_STRUCTURE.md
    └── LAPORAN_LENGKAP.md      # Full report (Indonesian)
```

---

## Research Background

### Problem Statement

Skin cancer is one of the most common types of cancer worldwide, with melanoma being the deadliest form. Early detection significantly improves survival rates, with 5-year survival rates exceeding 99% when detected early. However, access to dermatologists is limited in many regions, and manual screening is time-consuming and subject to inter-observer variability.

### Motivation

This project addresses the need for:

1. **Accessible Screening**: Automated preliminary assessment without requiring specialist consultation
2. **Scalability**: Handle large volumes of images efficiently
3. **Consistency**: Reduce subjective interpretation variance
4. **Speed**: Provide instant feedback for timely intervention
5. **Interpretability**: Explain model decisions for clinical trust

### Related Work

**Deep Learning in Dermatology:**
- Esteva et al. (2017) demonstrated dermatologist-level classification using CNNs
- HAM10000 dataset (Tschandl et al., 2018) established benchmarks for skin lesion analysis
- MobileNetV2 (Sandler et al., 2018) enables efficient deployment on resource-constrained devices

**Transfer Learning:**
- Pre-training on ImageNet provides rich visual features
- Fine-tuning adapts general features to medical domain
- Reduces need for large medical datasets

**Explainable AI:**
- Grad-CAM (Selvaraju et al., 2017) provides visual explanations
- Essential for clinical acceptance and trust
- Enables error analysis and model improvement

### Methodology

This implementation follows a systematic approach:

1. **Data Collection**: Curated dataset of dermatoscopic images
2. **Preprocessing**: Standardization and augmentation
3. **Model Selection**: MobileNetV2 for efficiency and accuracy
4. **Training**: Two-phase strategy with regularization
5. **Evaluation**: Comprehensive metrics and visualization
6. **Deployment**: User-friendly web interface
7. **Interpretation**: Grad-CAM for explainability

### Clinical Considerations

**Intended Use:**
- Preliminary screening tool
- Educational resource for medical students
- Research platform for ML in dermatology

**Limitations:**
- Not a replacement for professional diagnosis
- Performance may vary with image quality
- Limited to binary classification (benign vs malignant)
- Trained on specific dataset distribution

**Ethical Considerations:**
- Clear communication of limitations to users
- Recommendation for professional consultation
- Privacy protection for uploaded images
- Bias monitoring across different skin types

---

## Future Work

### Model Improvements

1. **Architecture Enhancement**
   - Experiment with EfficientNet, Vision Transformers
   - Ensemble methods combining multiple architectures
   - Progressive fine-tuning of base model layers

2. **Dataset Expansion**
   - Increase dataset size to 10,000+ images
   - Include multi-class classification (7+ skin lesion types)
   - Incorporate demographic metadata (age, skin type)
   - Add temporal data for lesion evolution tracking

3. **Regularization**
   - Advanced augmentation (MixUp, CutMix)
   - Test-time augmentation (TTA)
   - Focal loss for class imbalance
   - Additional dropout and L2 regularization

### Feature Development

4. **Segmentation**
   - Implement U-Net for lesion boundary detection
   - Extract ABCDE features (Asymmetry, Border, Color, Diameter, Evolution)
   - Automated lesion measurement

5. **Multi-Modal Learning**
   - Combine image data with patient history
   - Incorporate metadata (age, gender, location)
   - Temporal analysis of lesion changes

6. **Advanced Interpretability**
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Attention mechanism visualization

### Deployment

7. **Mobile Application**
   - React Native or Flutter app
   - Direct camera capture
   - Offline inference capability

8. **Cloud Deployment**
   - AWS Lambda / Google Cloud Run
   - Scalable API endpoint
   - Integration with telemedicine platforms

9. **Clinical Validation**
   - Pilot study with dermatology clinics
   - Prospective clinical trial
   - Regulatory approval (FDA, CE marking)

### Research Extensions

10. **Federated Learning**
    - Privacy-preserving collaborative training
    - Multi-institutional learning without data sharing

11. **Adversarial Robustness**
    - Testing against adversarial examples
    - Improving model reliability

12. **Uncertainty Quantification**
    - Bayesian neural networks
    - Confidence calibration
    - Out-of-distribution detection

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/YourFeature`
3. **Commit changes**: `git commit -m 'Add YourFeature'`
4. **Push to branch**: `git push origin feature/YourFeature`
5. **Open a Pull Request**

### Contribution Areas

- Bug fixes and issue resolution
- Documentation improvements
- New features and enhancements
- Performance optimization
- Test coverage expansion
- Dataset curation and validation

### Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for all functions
- Add unit tests for new features
- Update documentation accordingly
- Ensure backward compatibility

### Reporting Issues

Please use the GitHub issue tracker to report:
- Bugs and errors
- Feature requests
- Documentation gaps
- Performance problems

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 mfthsarsyd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See [LICENSE](LICENSE) file for full details.

---

## Citation

If you use this project in your research or application, please cite:

```bibtex
@software{skin_cancer_classification_2025,
  author = {mfthsarsyd},
  title = {Skin Cancer Classification using Deep Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Codift05/skin-cancer-classification-deep-learning}
}
```

### Academic References

**Key Papers:**

1. Esteva, A., et al. (2017). "Dermatologist-level classification of skin cancer with deep neural networks." *Nature*, 542(7639), 115-118.

2. Tschandl, P., et al. (2018). "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Scientific Data*, 5, 180161.

3. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*.

4. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." *ICCV*.

---

## Contact

**Author**: mfthsarsyd

**Project Link**: [https://github.com/Codift05/skin-cancer-classification-deep-learning](https://github.com/Codift05/skin-cancer-classification-deep-learning)

**Email**: [Contact via GitHub]

For questions, suggestions, or collaboration opportunities, please:
- Open an issue on GitHub
- Submit a pull request
- Contact via GitHub profile

---

## Acknowledgments

- **TensorFlow Team** for the deep learning framework
- **Streamlit** for the web application framework
- **ImageNet** for pre-trained weights
- **Dermatology Community** for medical insights
- **Open Source Contributors** for various tools and libraries

---

## Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This software is intended for research and educational purposes only. It is NOT a medical device and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.

**Key Points:**
- This tool provides preliminary screening only
- Results should NOT be used for clinical decision-making without professional verification
- Always consult with qualified healthcare professionals for medical concerns
- The developers assume no liability for any medical decisions based on this software
- Performance may vary based on image quality and patient population

If you have concerns about a skin lesion, please consult a board-certified dermatologist immediately.

---

**Copyright (c) 2025 mfthsarsyd. All Rights Reserved.**
