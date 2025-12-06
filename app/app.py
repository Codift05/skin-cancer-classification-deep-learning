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

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocess import load_and_prepare
from utils.helpers import (
    load_class_names, format_prediction, get_class_color, 
    get_class_emoji, get_model_info
)

# Import Grad-CAM utilities
from utils.gradcam import generate_gradcam, overlay_gradcam, get_last_conv_layer_name


# Configure Streamlit page
st.set_page_config(
    page_title="🏥 Skin Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 20px;
    }
    .success-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .danger-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
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


def main():
    # Header
    st.markdown("<h1 class='main-header'>🏥 Skin Cancer Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Deteksi kanker kulit menggunakan Deep Learning</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Pengaturan")
        
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
        
        st.success("✅ Model loaded successfully!")
        
        # Model information
        with st.expander("📊 Model Info"):
            info = get_model_info(model)
            st.metric("Total Parameters", f"{info['total_params']:,}")
            st.metric("Input Shape", f"{info['input_shape']}")
            st.write("**Architecture:** MobileNetV2 + Classification Head")
        
        # Settings
        st.markdown("---")
        show_gradcam = st.checkbox("🔥 Show Grad-CAM Heatmap", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.05, 0.01)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar kulit untuk dianalisis",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False
        )
    
    if uploaded_file is not None:
        # Load and display image
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        try:
            # Convert image to array
            img_array = np.array(image.convert('RGB'))
            
            # Resize to model input size using PIL
            image_resized = image.convert('RGB').resize((224, 224))
            img_resized = np.array(image_resized)
            
            # Normalize
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Make prediction
            with st.spinner("🔄 Analyzing image..."):
                prediction = model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0][0]
                result = format_prediction(prediction, class_names, confidence_threshold)
            
            # Display results
            with col2:
                st.markdown("### 🔍 Hasil Analisis")
                
                # Prediction result
                status_type, message = get_prediction_message(result)
                
                if status_type == "success":
                    st.markdown(f"<div class='success-box'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='danger-box'>{message}</div>", unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("#### Probability Distribution")
                for class_name, prob in result['all_probabilities'].items():
                    col_label, col_bar = st.columns([2, 8])
                    with col_label:
                        st.write(f"**{class_name.capitalize()}**")
                    with col_bar:
                        st.progress(float(prob) / 100.0)
                        st.write(f"{prob:.2f}%")
                
                # Warning for malignant
                if result['class'].lower() == 'malignant':
                    st.markdown("<div class='danger-box'>"
                              "⚠️ <strong>Konsultasi Dokter</strong><br/>"
                              "Hasil menunjukkan kemungkinan kanker kulit berbahaya. "
                              "Segera konsultasikan dengan dokter atau dermatolog."
                              "</div>", unsafe_allow_html=True)
                
                # Grad-CAM Visualization
                if show_gradcam:
                    st.markdown("---")
                    st.markdown("### 🔥 Grad-CAM Visualization")
                    st.info("Peta panas menunjukkan area yang paling mempengaruhi prediksi model (merah=penting)")
                    
                    with st.spinner("Generating Grad-CAM heatmap..."):
                        try:
                            # Get convolutional layer
                            layer_name = get_last_conv_layer_name(model)
                            st.success(f"Using layer: {layer_name}")
                            
                            # Generate Grad-CAM
                            heatmap = generate_gradcam(model, img_normalized, layer_name)
                            
                            # Apply colormap to heatmap for better visualization
                            import matplotlib
                            cmap_func = matplotlib.colormaps.get_cmap('jet')
                            heatmap_colored = cmap_func(heatmap)[:, :, :3]  # Get RGB, drop alpha
                            
                            overlaid = overlay_gradcam(img_normalized, heatmap, alpha=0.5)
                            
                            # Display visualizations
                            col_orig, col_heatmap, col_overlay = st.columns(3)
                            
                            with col_orig:
                                st.image(img_normalized, caption="Original Image", use_column_width=True)
                            
                            with col_heatmap:
                                st.image(heatmap_colored, caption="Grad-CAM Heatmap", use_column_width=True)
                            
                            with col_overlay:
                                st.image(overlaid, caption="Overlay", use_column_width=True)
                        
                        except Exception as e:
                            st.warning(f"⚠️ Grad-CAM generation failed: {str(e)}")
                            st.info("Model mungkin memiliki struktur yang tidak cocok untuk Grad-CAM")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    else:
        # Display instructions when no image is uploaded
        st.markdown("---")
        st.markdown("### 📋 Instruksi Penggunaan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Langkah-langkah:**
            1. Klik tombol "Browse Files" untuk memilih gambar
            2. Unggah foto kulit (JPG, PNG)
            3. Model akan menganalisis gambar
            4. Lihat hasil prediksi dan Grad-CAM
            
            **Catatan:**
            - Gunakan foto berkualitas tinggi
            - Pastikan area kulit terlihat jelas
            - Pencahayaan yang baik sangat penting
            """)
        
        with col2:
            st.markdown("""
            **Informasi:**
            - **Benign** ✅: Tidak berbahaya
            - **Malignant** ⚠️: Berbahaya, konsultasi dokter
            
            **Disclaimer:**
            ⚠️ Aplikasi ini adalah alat bantu diagnosis. 
            **Bukan pengganti konsultasi medis profesional.**
            
            Selalu konsultasikan dengan dokter dermatolog 
            untuk diagnosis definitif.
            """)
        
        st.markdown("---")
        
        # Example usage info
        with st.expander("ℹ️ Tentang Model"):
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
            
            **Visualization:**
            - Grad-CAM untuk memahami keputusan model
            - Heatmap menunjukkan area yang paling berpengaruh
            """)


if __name__ == "__main__":
    main()
