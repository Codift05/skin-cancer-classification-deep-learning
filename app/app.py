"""
Streamlit Web Application for Skin Cancer Classification
Interactive interface for skin cancer prediction with Grad-CAM visualization
"""

import streamlit as st
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image
import sys
import matplotlib.cm as cm
import base64

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocess import load_and_prepare
from utils.helpers import (
    load_class_names, format_prediction, get_class_color, 
    get_class_emoji, get_model_info
)

# Configure Streamlit page
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern flat design
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Styles */
    .main-header {
        text-align: center;
        color: #2c3e50;
        padding: 2rem 0;
        font-weight: 700;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Styles - Flat Design */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Success Box - Flat */
    .success-box {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #27ae60;
        color: white;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    /* Danger Box - Flat */
    .danger-box {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #e74c3c;
        color: white;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    /* Info Box - Flat */
    .info-box {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #3498db;
        color: white;
        margin: 1rem 0;
    }
    
    /* Warning Box - Flat */
    .warning-box {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #f39c12;
        color: white;
        margin: 1rem 0;
    }
    
    /* Icon Styles */
    .icon-svg {
        width: 24px;
        height: 24px;
        vertical-align: middle;
        margin-right: 8px;
    }
    
    /* Team Photo Placeholder */
    .team-photo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: white;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .team-photo h3 {
        margin: 0;
        font-size: 1.5rem;
    }
    
    .team-photo p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Upload Area */
    .upload-area {
        border: 3px dashed #95a5a6;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: white;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3498db;
        background: #f8f9fa;
    }
    
    /* Button Styles */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border: none;
        background: #3498db;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #2980b9;
        transform: translateY(-2px);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background-color: #3498db;
        border-radius: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Remove gradient from metrics */
    [data-testid="stMetricValue"] {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load and cache the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_resource
def load_labels(labels_path):
    """Load and cache class labels"""
    try:
        return load_class_names(labels_path)
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")
        return None


def get_prediction_message(result):
    """Format prediction result as message"""
    class_name = result['class']
    confidence = result['confidence']
    emoji = get_class_emoji(class_name)
    
    if class_name.lower() == 'benign':
        message = f"{emoji} **BENIGN** (Tidak berbahaya)\n\nKonfidansi: **{confidence:.2f}%**"
        return "success", message
    else:
        message = f"{emoji} **MALIGNANT** (Berbahaya - Konsultasi dokter)\n\nKonfidansi: **{confidence:.2f}%**"
        return "danger", message


def get_team_photo_base64():
    """Get team photo as base64 encoded string"""
    team_photo_path = Path(__file__).parent.parent / "assets" / "team.jpg"
    if team_photo_path.exists():
        with open(team_photo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def main():
    # Header with SVG Icon
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <svg class='icon-svg' style='width: 64px; height: 64px;' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>
            <path d='M12 2L2 7L12 12L22 7L12 2Z' fill='#3498db'/>
            <path d='M2 17L12 22L22 17' stroke='#3498db' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>
            <path d='M2 12L12 17L22 12' stroke='#3498db' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>
        </svg>
        <h1 class='main-header' style='margin: 1rem 0 0.5rem 0;'>Skin Cancer Classification</h1>
        <p class='subtitle'>Deteksi kanker kulit menggunakan Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model path selection
    model_path = Path(__file__).parent.parent / "model" / "skin_cancer_model_final.keras"
    labels_path = Path(__file__).parent.parent / "model" / "class_names.txt"
    
    # Check if files exist
    if not model_path.exists():
        st.error(f"❌ Model tidak ditemukan: {model_path}")
        st.stop()
    
    if not labels_path.exists():
        st.error(f"❌ Labels tidak ditemukan: {labels_path}")
        st.stop()
    
    # Load model and labels
    with st.spinner("Loading model..."):
        model = load_model(str(model_path))
        class_names = load_labels(str(labels_path))
    
    if model is None or class_names is None:
        st.stop()
    
    st.markdown("""
    <div class='success-box' style='background-color: #27ae60; padding: 1rem; border-radius: 8px; text-align: center;'>
        <svg style='width: 24px; height: 24px;' viewBox='0 0 24 24' fill='white' xmlns='http://www.w3.org/2000/svg'>
            <path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/>
        </svg>
        <span style='color: white; font-weight: 600;'>Model loaded successfully!</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    
    st.markdown("""
    <div class='card'>
        <svg class='icon-svg' style='width: 32px; height: 32px;' viewBox='0 0 24 24' fill='#3498db' xmlns='http://www.w3.org/2000/svg'>
            <path d='M9 2C7.89 2 7 2.89 7 4V20C7 21.11 7.89 22 9 22H18C19.11 22 20 21.11 20 20V8L14 2M13 3.5L18.5 9H13M11 11H13V13H15V15H13V17H11V15H9V13H11Z'/>
        </svg>
        <span style='font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin-left: 0.5rem;'>Upload Gambar</span>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih gambar kulit untuk dianalisis",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Load and display image in center
        image = Image.open(uploaded_file)
        
        # Center the image with max width
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process image
        try:
            # Convert image to array
            img_array = np.array(image.convert('RGB'))
            
            # Resize to model input size using PIL
            image_resized = image.convert('RGB').resize((224, 224))
            img_resized = np.array(image_resized)
            
            # Normalize
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Make prediction with modern loading
            with st.spinner("⚡ Analyzing image with AI..."):
                prediction = model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0][0]
                result = format_prediction(prediction, class_names, 0.05)
            
            # Display results in full width
            st.markdown("---")
            st.markdown("""
            <div class='card'>
                <svg class='icon-svg' style='width: 32px; height: 32px;' viewBox='0 0 24 24' fill='#3498db' xmlns='http://www.w3.org/2000/svg'>
                    <path d='M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2M18 20H6V4H13V9H18V20M16 11H8V13H16V11M16 15H8V17H16V15Z'/>
                </svg>
                <span style='font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin-left: 0.5rem;'>Hasil Analisis</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Prediction result
            status_type, message = get_prediction_message(result)
            
            # Modern flat result card
            if status_type == "success":
                class_name = result['class']
                confidence = result['confidence']
                st.markdown(f"""
                <div class='success-box' style='text-align: center;'>
                    <svg style='width: 48px; height: 48px; margin-bottom: 1rem;' viewBox='0 0 24 24' fill='white' xmlns='http://www.w3.org/2000/svg'>
                        <path d='M12 2C6.5 2 2 6.5 2 12S6.5 22 12 22 22 17.5 22 12 17.5 2 12 2M10 17L5 12L6.41 10.59L10 14.17L17.59 6.58L19 8L10 17Z'/>
                    </svg>
                    <h2 style='margin: 0; font-size: 1.8rem; font-weight: 700; color: white;'>{class_name.upper()}</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95;'>Tidak berbahaya</p>
                    <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);'>
                        <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Confidence</p>
                        <p style='margin: 0.3rem 0 0 0; font-size: 2rem; font-weight: 700;'>{confidence:.2f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                class_name = result['class']
                confidence = result['confidence']
                st.markdown(f"""
                <div class='danger-box' style='text-align: center;'>
                    <svg style='width: 48px; height: 48px; margin-bottom: 1rem;' viewBox='0 0 24 24' fill='white' xmlns='http://www.w3.org/2000/svg'>
                        <path d='M12 2C6.5 2 2 6.5 2 12S6.5 22 12 22 22 17.5 22 12 17.5 2 12 2M13 17H11V15H13V17M13 13H11V7H13V13Z'/>
                    </svg>
                    <h2 style='margin: 0; font-size: 1.8rem; font-weight: 700; color: white;'>{class_name.upper()}</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95;'>Berbahaya - Konsultasi dokter</p>
                    <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);'>
                        <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Confidence</p>
                        <p style='margin: 0.3rem 0 0 0; font-size: 2rem; font-weight: 700;'>{confidence:.2f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability breakdown - Modern flat design
            st.markdown("""
            <div style='margin: 2rem 0 1rem 0;'>
                <h4 style='color: #2c3e50; font-weight: 600; font-size: 1.1rem;'>
                    <svg style='width: 24px; height: 24px; vertical-align: middle; margin-right: 8px;' viewBox='0 0 24 24' fill='#3498db' xmlns='http://www.w3.org/2000/svg'>
                        <path d='M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3M9 17H7V10H9V17M13 17H11V7H13V17M17 17H15V13H17V17Z'/>
                    </svg>
                    Probability Distribution
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display probabilities in modern cards
            for class_name, prob in result['all_probabilities'].items():
                # Color based on class
                if class_name.lower() == 'benign':
                    color = '#27ae60'
                    icon_path = 'M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'
                else:
                    color = '#e74c3c'
                    icon_path = 'M13 14H11V9H13M13 18H11V16H13M1 21H23L12 2L1 21Z'
                
                st.markdown(f"""
                <div style='background: white; border-left: 4px solid {color}; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='display: flex; align-items: center;'>
                            <svg style='width: 20px; height: 20px; margin-right: 0.5rem;' viewBox='0 0 24 24' fill='{color}' xmlns='http://www.w3.org/2000/svg'>
                                <path d='{icon_path}'/>
                            </svg>
                            <span style='font-weight: 600; color: #2c3e50; font-size: 1rem;'>{class_name.capitalize()}</span>
                        </div>
                        <span style='font-weight: 700; color: {color}; font-size: 1.2rem;'>{prob:.2f}%</span>
                    </div>
                    <div style='background: #ecf0f1; border-radius: 10px; height: 8px; margin-top: 0.8rem; overflow: hidden;'>
                        <div style='background: {color}; height: 100%; width: {prob}%; border-radius: 10px; transition: width 0.3s ease;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Warning for malignant
            if result['class'].lower() == 'malignant':
                st.markdown("""
                <div class='warning-box'>
                    <svg style='width: 28px; height: 28px; vertical-align: middle;' viewBox='0 0 24 24' fill='white' xmlns='http://www.w3.org/2000/svg'>
                        <path d='M13 14H11V9H13M13 18H11V16H13M1 21H23L12 2L1 21Z'/>
                    </svg>
                    <strong>Konsultasi Dokter</strong><br/>
                    Hasil menunjukkan kemungkinan kanker kulit berbahaya. 
                    Segera konsultasikan dengan dokter atau dermatolog.
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    else:
        # Display instructions when no image is uploaded
        st.markdown("---")
        
        # Team Photo Section with Medical Theme
        team_photo_b64 = get_team_photo_base64()
        
        if team_photo_b64:
            # Use local image
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 2rem; text-align: center; color: white; min-height: 350px; display: flex; flex-direction: column; justify-content: center; align-items: center; background-image: url("data:image/jpeg;base64,{team_photo_b64}"); background-size: cover; background-position: center; position: relative;'>
                <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(135deg, rgba(102, 126, 234, 0.75) 0%, rgba(118, 75, 162, 0.75) 100%); border-radius: 12px;'></div>
                <div style='position: relative; z-index: 1;'>
                    <h3 style='margin: 0; font-size: 2rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>Medical Team</h3>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>Powered by AI & Medical Expertise</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback placeholder
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 2rem; text-align: center; color: white; min-height: 350px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                <svg style='width: 80px; height: 80px; margin-bottom: 1rem;' viewBox='0 0 24 24' fill='white' xmlns='http://www.w3.org/2000/svg'>
                    <path d='M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3M19 19H5V5H19V19M13.96 12.29L11.21 15.83L9.25 13.47L6.5 17H17.5L13.96 12.29Z'/>
                </svg>
                <h3 style='margin: 0; font-size: 1.8rem; font-weight: 700;'></h3>
                <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95;'>Add team_photo.jpg to assets folder</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <svg class='icon-svg' style='width: 32px; height: 32px;' viewBox='0 0 24 24' fill='#3498db' xmlns='http://www.w3.org/2000/svg'>
                <path d='M19 3H14.82C14.4 1.84 13.3 1 12 1S9.6 1.84 9.18 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3M12 3C12.55 3 13 3.45 13 4S12.55 5 12 5 11 4.55 11 4 11.45 3 12 3M7 7H17V5H19V19H5V5H7V7M12 17L17 12L15.59 10.59L12 14.17L9.41 11.59L8 13L12 17Z'/>
            </svg>
            <span style='font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin-left: 0.5rem;'>Instruksi Penggunaan</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='card'>
                <h4 style='color: #2c3e50;'>
                    <svg class='icon-svg' viewBox='0 0 24 24' fill='#3498db' xmlns='http://www.w3.org/2000/svg'>
                        <path d='M9 5V9H21V5M9 19H21V15H9M9 14H21V10H9M4 9H8V5H4M4 19H8V15H4M4 14H8V10H4Z'/>
                    </svg>
                    Langkah-langkah:
                </h4>
                <ol style='color: #34495e; line-height: 2;'>
                    <li>Klik tombol "Browse Files" untuk memilih gambar</li>
                    <li>Unggah foto kulit (JPG, PNG)</li>
                    <li>Model akan menganalisis gambar</li>
                    <li>Lihat hasil prediksi secara real-time</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='card'>
                <h4 style='color: #2c3e50;'>
                    <svg class='icon-svg' viewBox='0 0 24 24' fill='#3498db' xmlns='http://www.w3.org/2000/svg'>
                        <path d='M13 9H11V7H13M13 17H11V11H13M12 2A10 10 0 0 0 2 12A10 10 0 0 0 12 22A10 10 0 0 0 22 12A10 10 0 0 0 12 2Z'/>
                    </svg>
                    Informasi:
                </h4>
                <div style='margin: 1rem 0;'>
                    <div style='padding: 0.8rem; background: #27ae60; color: white; border-radius: 8px; margin-bottom: 0.5rem;'>
                        <svg style='width: 20px; height: 20px; vertical-align: middle;' viewBox='0 0 24 24' fill='white' xmlns='http://www.w3.org/2000/svg'>
                            <path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/>
                        </svg>
                        <strong>Benign:</strong> Tidak berbahaya
                    </div>
                    <div style='padding: 0.8rem; background: #e74c3c; color: white; border-radius: 8px;'>
                        <svg style='width: 20px; height: 20px; vertical-align: middle;' viewBox='0 0 24 24' fill='white' xmlns='http://www.w3.org/2000/svg'>
                            <path d='M13 14H11V9H13M13 18H11V16H13M1 21H23L12 2L1 21Z'/>
                        </svg>
                        <strong>Malignant:</strong> Berbahaya, konsultasi dokter
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Example usage info
        with st.expander("Tentang Model"):
            st.markdown("""
            **Model:** MobileNetV2 with Transfer Learning
            - **Architecture:** Pre-trained MobileNetV2 + Custom Classification Head
            - **Input Size:** 224x224x3
            - **Classes:** Benign, Malignant (Binary Classification)
            - **Framework:** TensorFlow/Keras
            
            **Training Details:**
            - Optimizer: Adam (lr=1e-4)
            - Loss: Binary Crossentropy
            - Callbacks: EarlyStopping, ReduceLROnPlateau
            
            **Performance:**
            - Binary classification: Benign vs Malignant
            - Real-time prediction with confidence scores
            """)


if __name__ == "__main__":
    main()
