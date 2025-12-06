"""
Helper utilities for Skin Cancer Classification
Useful functions for model management and prediction
"""

import numpy as np
from pathlib import Path


def load_class_names(labels_file):
    """
    Load class names from labels file
    
    Args:
        labels_file (str or Path): Path to labels.txt file
    
    Returns:
        list: List of class names
    """
    labels_file = Path(labels_file)
    
    with open(labels_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return class_names


def format_prediction(prediction_prob, class_names, threshold=0.05):
    """
    Format prediction output for display
    
    Args:
        prediction_prob (float): Prediction probability (0-1)
        class_names (list): List of class names
        threshold (float): Classification threshold (default 0.05 for optimal balance)
    
    Returns:
        dict: Formatted prediction result
    """
    predicted_class_idx = int(prediction_prob > threshold)
    predicted_class = class_names[predicted_class_idx]
    
    # Calculate confidence
    if predicted_class_idx == 1:  # Positive class
        confidence = prediction_prob * 100
    else:  # Negative class
        confidence = (1 - prediction_prob) * 100
    
    return {
        'class': predicted_class,
        'class_index': predicted_class_idx,
        'confidence': confidence,
        'probability': prediction_prob,
        'all_probabilities': {
            class_names[0]: (1 - prediction_prob) * 100,
            class_names[1]: prediction_prob * 100
        }
    }


def get_class_color(class_name):
    """
    Get color for class for visualization
    
    Args:
        class_name (str): Class name (e.g., 'benign', 'malignant')
    
    Returns:
        tuple: RGB color tuple or Hex color code
    """
    class_colors = {
        'benign': '#00AA00',      # Green
        'malignant': '#FF0000',   # Red
    }
    
    return class_colors.get(class_name.lower(), '#0000FF')


def get_class_emoji(class_name):
    """
    Get emoji for class
    
    Args:
        class_name (str): Class name
    
    Returns:
        str: Emoji representation
    """
    class_emojis = {
        'benign': '✅',
        'malignant': '⚠️',
    }
    
    return class_emojis.get(class_name.lower(), '❓')


def validate_image_path(image_path):
    """
    Validate if image path exists and has valid extension
    
    Args:
        image_path (str or Path): Path to image file
    
    Returns:
        bool: True if valid, False otherwise
    """
    image_path = Path(image_path)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if not image_path.exists():
        return False
    
    if image_path.suffix.lower() not in valid_extensions:
        return False
    
    return True


def get_model_info(model):
    """
    Get information about a Keras model
    
    Args:
        model: Keras model
    
    Returns:
        dict: Model information
    """
    return {
        'total_params': model.count_params(),
        'trainable_params': np.sum([np.prod(w.shape) for w in model.trainable_weights]),
        'non_trainable_params': np.sum([np.prod(w.shape) for w in model.non_trainable_weights]),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
    }


def batch_predict(model, images_array, batch_size=32):
    """
    Make predictions on batch of images
    
    Args:
        model: Keras model
        images_array (np.ndarray): Batch of images
        batch_size (int): Batch size for prediction
    
    Returns:
        np.ndarray: Predictions
    """
    predictions = []
    
    for i in range(0, len(images_array), batch_size):
        batch = images_array[i:i+batch_size]
        batch_pred = model.predict(batch, verbose=0)
        predictions.extend(batch_pred)
    
    return np.array(predictions)


def calculate_statistics(predictions, true_labels=None):
    """
    Calculate statistics from predictions
    
    Args:
        predictions (np.ndarray): Model predictions
        true_labels (np.ndarray, optional): True labels for comparison
    
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'mean_prediction': float(np.mean(predictions)),
        'std_prediction': float(np.std(predictions)),
        'min_prediction': float(np.min(predictions)),
        'max_prediction': float(np.max(predictions)),
        'median_prediction': float(np.median(predictions)),
    }
    
    if true_labels is not None:
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        accuracy = np.mean(predicted_classes == true_labels)
        stats['accuracy'] = float(accuracy)
    
    return stats


if __name__ == "__main__":
    print("Helper utilities loaded successfully!")
