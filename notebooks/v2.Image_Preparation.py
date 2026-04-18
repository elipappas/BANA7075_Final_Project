import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

DATA_DIR = Path("data")                    
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)


val_test_datagen = ImageDataGenerator(rescale=1./255)


def get_training_set():
    train_path = DATA_DIR / "train"
    if not train_path.exists():
        raise FileNotFoundError(
            f"❌ Training directory not found: {train_path}\n"
            f"Please download the Chest X-Ray Pneumonia dataset from Kaggle "
            f"and place it under the '{DATA_DIR}' folder (see README.md for details)."
        )
    
    return train_datagen.flow_from_directory(
        directory=str(train_path),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )


def get_validation_set():
    val_path = DATA_DIR / "val"
    if not val_path.exists():
        raise FileNotFoundError(f"❌ Validation directory not found: {val_path}")
    
    return val_test_datagen.flow_from_directory(
        directory=str(val_path),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )


def get_test_set():
    test_path = DATA_DIR / "test"
    if not test_path.exists():
        raise FileNotFoundError(f"❌ Test directory not found: {test_path}")
    
    return val_test_datagen.flow_from_directory(
        directory=str(test_path),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
