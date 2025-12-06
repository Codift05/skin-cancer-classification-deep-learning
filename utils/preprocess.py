"""
Preprocessing utilities for Skin Cancer Classification
Handles image loading, normalization, and preparation
"""

import numpy as np
from PIL import Image
from pathlib import Path


def load_image(image_path, target_size=(224, 224)):
    """
    Load and resize image to target size
    
    Args:
        image_path (str or Path): Path to the image file
        target_size (tuple): Target size (height, width) for resizing
    
    Returns:
        np.ndarray: Image array with shape (height, width, 3)
    """
    try:
        # Load image using PIL
        img = Image.open(image_path).convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Error loading image from {image_path}: {str(e)}")


def prepare(img_array, normalize=True):
    """
    Prepare image array for model prediction
    - Normalize pixel values to 0-1 range
    - Ensure correct shape
    
    Args:
        img_array (np.ndarray): Input image array with shape (224, 224, 3)
        normalize (bool): Whether to normalize pixel values to 0-1
    
    Returns:
        np.ndarray: Prepared image array ready for model prediction
    """
    # Ensure image is in correct format
    if len(img_array.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {img_array.shape}")
    
    if img_array.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got {img_array.shape[2]}")
    
    # Normalize pixel values
    if normalize:
        if img_array.max() > 1:
            # If values are 0-255, normalize to 0-1
            img_array = img_array.astype(np.float32) / 255.0
        else:
            # Already normalized
            img_array = img_array.astype(np.float32)
    
    return img_array


def load_and_prepare(image_path, target_size=(224, 224), normalize=True):
    """
    Load and prepare image in one step
    
    Args:
        image_path (str or Path): Path to the image file
        target_size (tuple): Target size (height, width) for resizing
        normalize (bool): Whether to normalize pixel values to 0-1
    
    Returns:
        np.ndarray: Ready-to-use image array for model prediction
    """
    img_array = load_image(image_path, target_size)
    img_array = prepare(img_array, normalize)
    return img_array


def batch_load_images(image_dir, target_size=(224, 224), normalize=True, recursive=True):
    """
    Load multiple images from a directory
    
    Args:
        image_dir (str or Path): Directory containing images
        target_size (tuple): Target size (height, width) for resizing
        normalize (bool): Whether to normalize pixel values to 0-1
        recursive (bool): Whether to search subdirectories
    
    Returns:
        tuple: (images list, file names list)
    """
    image_dir = Path(image_dir)
    images = []
    filenames = []
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Find all images
    if recursive:
        image_files = [f for f in image_dir.rglob('*') 
                      if f.suffix.lower() in image_extensions]
    else:
        image_files = [f for f in image_dir.glob('*') 
                      if f.suffix.lower() in image_extensions]
    
    for image_path in sorted(image_files):
        try:
            img_array = load_and_prepare(image_path, target_size, normalize)
            images.append(img_array)
            filenames.append(image_path.name)
        except Exception as e:
            print(f"Warning: Skipped {image_path} - {str(e)}")
    
    return np.array(images), filenames


if __name__ == "__main__":
    print("Preprocessing utilities loaded successfully!")
