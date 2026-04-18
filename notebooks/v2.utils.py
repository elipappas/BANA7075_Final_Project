
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array



IMAGE_SIZE = (224, 224)


def prepare_image_for_model(image_input, target_size=None):
    """
    Preprocess a single image for model inference.
    
    Args:
        image_input: Can be a file path (str), bytes, or PIL Image object.
        target_size: Optional override (defaults to (224, 224) to match training).
    
    Returns:
        Numpy array of shape (1, 224, 224, 3) with values in [0, 1].
    """
    if target_size is None:
        target_size = IMAGE_SIZE

  
    if isinstance(image_input, (str, bytes)) or hasattr(image_input, 'read'):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("Input must be a file path, file-like object, or PIL Image.")

   
    if img.mode != 'RGB':
        img = img.convert('RGB')

    
    img = img.resize(target_size)

    
    img_array = img_to_array(img)

  
    img_array = np.expand_dims(img_array, axis=0)

 
    img_array = img_array / 255.0

    return img_array
