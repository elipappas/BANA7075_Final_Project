import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

def get_training_set():
    training_set = train_datagen.flow_from_directory(
        './data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    return training_set

def get_validation_set():
    # Validation data (no augmentation, only rescaling)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    validation_set = valid_datagen.flow_from_directory(
        './data/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    return validation_set

def get_test_set():
    # Test data (no augmentation, no shuffle!)
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_set = test_datagen.flow_from_directory(
        './data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    return test_set