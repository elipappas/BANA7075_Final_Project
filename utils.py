import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def prepare_image_for_model(image_input, target_size=(224, 224)):
    # Load the image if it's a path or file-like object (like Streamlit upload)
    if isinstance(image_input, (str, bytes)) or hasattr(image_input, 'read'):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("Input must be a file path, file-like object, or PIL Image.")

    # Convert to RGB (ensures exactly 3 color channels, drops alpha/transparency)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize to the target size expected by the model
    img = img.resize(target_size)
    
    # Convert PIL image to a mathematical NumPy array
    img_array = img_to_array(img)
    
    # Expand dimensions to create a "batch" of 1: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale pixel values (Crucial: Matches your valid/test ImageDataGenerator)
    img_array = img_array / 255.0
    
    return img_array