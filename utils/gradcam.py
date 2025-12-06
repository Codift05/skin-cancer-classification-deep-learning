"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
For visualizing model predictions and understanding model decision-making
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_gradcam(model, img_array, layer_name, colormap_name='jet'):
    """
    Generate Grad-CAM heatmap
    
    Args:
        model: Trained Keras model
        img_array: Input image array (224, 224, 3) - normalized [0, 1]
        layer_name: Name of convolutional layer
        colormap_name: Matplotlib colormap name (not used in function but kept for API compatibility)
    
    Returns:
        np.ndarray: Grad-CAM heatmap (224, 224) with values in range [0, 1]
    """
    try:
        # Ensure correct input shape
        if len(img_array.shape) == 3:
            img_input = np.expand_dims(img_array, axis=0)
        else:
            img_input = img_array
        
        img_input = tf.constant(img_input, dtype=tf.float32)
        
        # Find appropriate layer if not provided
        if layer_name is None:
            layer_name = get_last_conv_layer_name(model)
        
        # Watch the input and get layer output
        with tf.GradientTape() as tape:
            # Get intermediate layer output
            last_conv_layer = model.get_layer(layer_name)
            iterate = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
            model_out, last_conv_layer_output = iterate(img_input)
            
            # Get the predicted class score
            class_out = model_out[:, 0]
        
        # Compute gradients of class output with respect to feature map
        grads = tape.gradient(class_out, last_conv_layer_output)
        
        if grads is None:
            print("Gradients are None!")
            return np.zeros((224, 224))
        
        # Global average pooling to get importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get values
        last_conv_layer_output = last_conv_layer_output[0]
        
        # Multiply each channel by its importance weight
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU on top of the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize to image size
        heatmap = np.uint8(255 * heatmap)
        
        # Use PIL to resize
        from PIL import Image
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize((224, 224), Image.Resampling.BILINEAR)
        heatmap = np.array(heatmap)
        
        # Normalize back to [0, 1]
        heatmap = heatmap / 255.0
        
        return heatmap
    
    except Exception as e:
        print(f"Grad-CAM error: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.zeros((224, 224))


def overlay_gradcam(img_array, heatmap, alpha=0.4, colormap_name='jet'):
    """
    Overlay Grad-CAM heatmap on original image using PIL and matplotlib
    
    Args:
        img_array (np.ndarray): Original image (224, 224, 3), normalized [0, 1]
        heatmap (np.ndarray): Grad-CAM heatmap (224, 224)
        alpha (float): Transparency of heatmap overlay (0-1)
        colormap_name (str): Matplotlib colormap name ('jet', 'hot', 'cool', 'bone', 'viridis')
    
    Returns:
        np.ndarray: Image with overlaid heatmap (uint8, 0-255 range)
    """
    # Ensure heatmap is 2D
    if len(heatmap.shape) != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")
    
    # Resize heatmap to match image size if needed using PIL
    if heatmap.shape != (224, 224):
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((224, 224), Image.Resampling.BILINEAR)
        heatmap = np.array(heatmap_img) / 255.0
    
    # Convert heatmap to 0-255 uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply matplotlib colormap
    cmap = cm.get_cmap(colormap_name)
    heatmap_colored = cmap(heatmap)  # Returns RGBA (0-1 range)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)  # Get RGB, convert to 0-255
    
    # Convert image to 0-255 range if needed
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
    
    # Blend images using numpy
    img_float = img_uint8.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    overlaid = (img_float * (1 - alpha) + heatmap_float * alpha).astype(np.uint8)
    
    return overlaid


def generate_gradcam_with_overlay(model, img_array, layer_name, alpha=0.4, colormap_name='jet'):
    """
    Generate both heatmap and overlaid visualization in one step
    
    Args:
        model: Trained Keras model
        img_array: Input image array (normalized, shape: (224, 224, 3))
        layer_name: Name of convolutional layer
        alpha: Transparency of overlay
        colormap_name: Matplotlib colormap name
    
    Returns:
        tuple: (heatmap, overlaid_image)
    """
    heatmap = generate_gradcam(model, img_array, layer_name)
    overlaid = overlay_gradcam(img_array, heatmap, alpha, colormap_name)
    return heatmap, overlaid


def save_gradcam_visualization(model, img_array, layer_name, save_path, alpha=0.4):
    """
    Generate and save Grad-CAM visualization
    
    Args:
        model: Trained Keras model
        img_array: Input image array (normalized)
        layer_name: Convolutional layer name
        save_path: Path to save the visualization
        alpha: Transparency of overlay
    """
    heatmap = generate_gradcam(model, img_array, layer_name)
    overlaid = overlay_gradcam(img_array, heatmap, alpha)
    
    # Save using PIL
    overlaid_img = Image.fromarray(overlaid)
    overlaid_img.save(str(save_path))
    
    print(f"Grad-CAM visualization saved to: {save_path}")

def get_last_conv_layer_name(model):
    """
    Find the last layer with actual feature maps (Conv2D or similar)
    Skip Dense, Dropout, and other layers
    
    Args:
        model: Keras model
    
    Returns:
        str: Name of a convolutional/pooling layer with feature maps
    """
    # Look for layers that have 4D output (batch, height, width, channels)
    for layer in reversed(model.layers):
        layer_type = type(layer).__name__
        
        # Skip these layer types - they don't have spatial feature maps
        if layer_type in ['Dense', 'Dropout', 'Flatten', 'Input', 'InputLayer']:
            continue
        
        # Check if layer has output shape with 4 dimensions
        try:
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
                # Check if 4D (batch, height, width, channels)
                if isinstance(output_shape, tuple) and len(output_shape) == 4:
                    return layer.name
        except:
            pass
    
    # Fallback: return any non-input, non-dense layer
    for layer in reversed(model.layers):
        layer_type = type(layer).__name__
        if layer_type not in ['Dense', 'InputLayer', 'Input']:
            return layer.name
    
    # Last resort
    return model.layers[-2].name


# Colormap options (matplotlib colormaps)
COLORMAPS = {
    'jet': 'jet',
    'hot': 'hot',
    'cool': 'cool',
    'bone': 'bone',
    'viridis': 'viridis',
}


if __name__ == "__main__":
    print("Grad-CAM utilities loaded successfully!")

