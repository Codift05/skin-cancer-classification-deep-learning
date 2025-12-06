# 📚 API Documentation - Utils

Complete API reference untuk utility modules.

## Table of Contents
- [preprocess.py](#preprocesspy)
- [gradcam.py](#gradcampy)
- [helpers.py](#helperspy)

---

## preprocess.py

Image preprocessing dan preparation functions.

### `load_image(image_path, target_size=(224, 224))`

Load dan resize image.

**Parameters:**
- `image_path` (str or Path): Path ke file image
- `target_size` (tuple): Target size (height, width) untuk resize

**Returns:**
- `np.ndarray`: Image array dengan shape (224, 224, 3)

**Example:**
```python
from utils.preprocess import load_image

img = load_image('path/to/image.jpg')
print(img.shape)  # Output: (224, 224, 3)
```

---

### `prepare(img_array, normalize=True)`

Prepare image array untuk model prediction.

**Parameters:**
- `img_array` (np.ndarray): Input image dengan shape (224, 224, 3)
- `normalize` (bool): Normalize pixel values ke 0-1 range

**Returns:**
- `np.ndarray`: Prepared image array (float32, normalized)

**Example:**
```python
img_prepared = prepare(img, normalize=True)
# img_prepared ready untuk model.predict()
```

---

### `load_and_prepare(image_path, target_size=(224, 224), normalize=True)`

Load dan prepare image dalam satu step.

**Parameters:**
- `image_path` (str or Path): Path ke image file
- `target_size` (tuple): Target size untuk resize
- `normalize` (bool): Normalisasi pixel values

**Returns:**
- `np.ndarray`: Ready-to-use image array

**Example:**
```python
from utils.preprocess import load_and_prepare

img = load_and_prepare('path/to/image.jpg')
predictions = model.predict(np.expand_dims(img, axis=0))
```

---

### `batch_load_images(image_dir, target_size=(224, 224), normalize=True, recursive=True)`

Load multiple images dari direktori.

**Parameters:**
- `image_dir` (str or Path): Direktori berisi images
- `target_size` (tuple): Target size untuk resize
- `normalize` (bool): Normalisasi pixel values
- `recursive` (bool): Search subdirectories

**Returns:**
- `tuple`: (images array, filenames list)

**Example:**
```python
images, filenames = batch_load_images('data/benign', recursive=False)
print(f"Loaded {len(images)} images")
```

---

## gradcam.py

Grad-CAM visualization untuk model interpretability.

### `generate_gradcam(model, img_array, layer_name)`

Generate Grad-CAM heatmap.

**Parameters:**
- `model`: Keras model
- `img_array` (np.ndarray): Input image (normalized, shape: 224, 224, 3)
- `layer_name` (str): Convolutional layer name untuk visualization

**Returns:**
- `np.ndarray`: Grad-CAM heatmap (224, 224) dengan values 0-1

**Example:**
```python
from utils.gradcam import generate_gradcam

heatmap = generate_gradcam(model, img_array, 'mobilenetv2_1_out_relu')
```

---

### `overlay_gradcam(img_array, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET)`

Overlay Grad-CAM heatmap pada original image.

**Parameters:**
- `img_array` (np.ndarray): Original image (224, 224, 3), normalized 0-1
- `heatmap` (np.ndarray): Grad-CAM heatmap (224, 224)
- `alpha` (float): Transparency 0-1 (0=transparent, 1=opaque)
- `colormap` (int): OpenCV colormap code

**Returns:**
- `np.ndarray`: Overlaid image (uint8, 0-255 range)

**Example:**
```python
overlaid = overlay_gradcam(img_array, heatmap, alpha=0.4)
plt.imshow(overlaid)
plt.show()
```

---

### `generate_gradcam_with_overlay(model, img_array, layer_name, alpha=0.4, colormap=cv2.COLORMAP_JET)`

Generate heatmap dan overlay dalam satu step.

**Parameters:**
- `model`: Keras model
- `img_array`: Input image
- `layer_name`: Convolutional layer name
- `alpha`: Overlay transparency
- `colormap`: OpenCV colormap

**Returns:**
- `tuple`: (heatmap, overlaid_image)

**Example:**
```python
heatmap, overlaid = generate_gradcam_with_overlay(
    model, img_array, 'mobilenetv2_1_out_relu'
)
```

---

### `save_gradcam_visualization(model, img_array, layer_name, save_path, alpha=0.4)`

Generate dan save Grad-CAM visualization.

**Parameters:**
- `model`: Keras model
- `img_array`: Input image
- `layer_name`: Convolutional layer name
- `save_path` (str or Path): Path untuk save image
- `alpha`: Overlay transparency

**Example:**
```python
save_gradcam_visualization(
    model, img_array, 
    'mobilenetv2_1_out_relu',
    'output/gradcam.png'
)
```

---

### `get_last_conv_layer_name(model)`

Automatically find last convolutional layer name.

**Parameters:**
- `model`: Keras model

**Returns:**
- `str`: Layer name

**Example:**
```python
layer = get_last_conv_layer_name(model)
heatmap = generate_gradcam(model, img, layer)
```

---

## helpers.py

Helper utilities untuk model management dan prediction.

### `load_class_names(labels_file)`

Load class names dari file.

**Parameters:**
- `labels_file` (str or Path): Path ke labels.txt file

**Returns:**
- `list`: List of class names

**Example:**
```python
from utils.helpers import load_class_names

classes = load_class_names('model/labels.txt')
# ['benign', 'malignant']
```

---

### `format_prediction(prediction_prob, class_names, threshold=0.5)`

Format prediction output.

**Parameters:**
- `prediction_prob` (float): Prediction probability 0-1
- `class_names` (list): List of class names
- `threshold` (float): Classification threshold

**Returns:**
- `dict`: Formatted prediction result

**Example:**
```python
result = format_prediction(0.85, ['benign', 'malignant'])
# {
#     'class': 'malignant',
#     'class_index': 1,
#     'confidence': 85.0,
#     'probability': 0.85,
#     'all_probabilities': {'benign': 15.0, 'malignant': 85.0}
# }
```

---

### `get_class_color(class_name)`

Get color untuk class visualization.

**Parameters:**
- `class_name` (str): Class name

**Returns:**
- `str`: Hex color code

**Example:**
```python
color = get_class_color('benign')    # '#00AA00'
color = get_class_color('malignant')  # '#FF0000'
```

---

### `get_class_emoji(class_name)`

Get emoji untuk class.

**Parameters:**
- `class_name` (str): Class name

**Returns:**
- `str`: Emoji

**Example:**
```python
emoji = get_class_emoji('benign')    # '✅'
emoji = get_class_emoji('malignant')  # '⚠️'
```

---

### `validate_image_path(image_path)`

Validate image path.

**Parameters:**
- `image_path` (str or Path): Path to image

**Returns:**
- `bool`: True jika valid, False sebaliknya

**Example:**
```python
is_valid = validate_image_path('image.jpg')
```

---

### `get_model_info(model)`

Get model information.

**Parameters:**
- `model`: Keras model

**Returns:**
- `dict`: Model info dictionary

**Example:**
```python
info = get_model_info(model)
# {
#     'total_params': 2250000,
#     'trainable_params': 128000,
#     'non_trainable_params': 2122000,
#     'layers': 154,
#     'input_shape': (None, 224, 224, 3),
#     'output_shape': (None, 1)
# }
```

---

### `batch_predict(model, images_array, batch_size=32)`

Make predictions on batch of images.

**Parameters:**
- `model`: Keras model
- `images_array` (np.ndarray): Batch of images
- `batch_size` (int): Batch size

**Returns:**
- `np.ndarray`: Predictions

**Example:**
```python
predictions = batch_predict(model, images_batch, batch_size=32)
```

---

### `calculate_statistics(predictions, true_labels=None)`

Calculate statistics dari predictions.

**Parameters:**
- `predictions` (np.ndarray): Model predictions
- `true_labels` (np.ndarray, optional): True labels

**Returns:**
- `dict`: Statistics dictionary

**Example:**
```python
stats = calculate_statistics(predictions, true_labels)
# {
#     'mean_prediction': 0.65,
#     'std_prediction': 0.18,
#     'min_prediction': 0.02,
#     'max_prediction': 0.99,
#     'median_prediction': 0.68,
#     'accuracy': 0.92
# }
```

---

## Usage Examples

### Complete Prediction Pipeline

```python
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import load_and_prepare
from utils.helpers import load_class_names, format_prediction
from utils.gradcam import generate_gradcam, overlay_gradcam

# 1. Load model & labels
model = load_model('model/model.h5')
class_names = load_class_names('model/labels.txt')

# 2. Load & prepare image
img = load_and_prepare('path/to/image.jpg')

# 3. Make prediction
prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
result = format_prediction(prediction, class_names)

# 4. Display results
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}%")

# 5. Generate Grad-CAM
heatmap = generate_gradcam(model, img, 'mobilenetv2_1_out_relu')
overlaid = overlay_gradcam(img, heatmap)
```

### Batch Prediction

```python
from utils.preprocess import batch_load_images
from utils.helpers import batch_predict

# Load images
images, filenames = batch_load_images('data/benign')

# Predict
predictions = batch_predict(model, images)

# Process results
for filename, pred in zip(filenames, predictions):
    class_idx = int(pred > 0.5)
    print(f"{filename}: {class_names[class_idx]} ({pred[0]*100:.2f}%)")
```

---

**Last Updated:** December 2024
