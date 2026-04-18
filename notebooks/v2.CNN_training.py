import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.keras
import os


from Image_Preparation import (
    get_training_set,
    get_validation_set,
    get_test_set
)


EPOCHS = 5
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_SAVE_PATH = "xray_cnn_model.keras"


tf.random.set_seed(42)


mlflow.set_experiment("Pneumonia_Detection_CNN")

def build_model():
    """Build the CNN model architecture."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    print("🚀 Starting Pneumonia Detection CNN Training...\n")
    
    
    train_generator = get_training_set()
    val_generator = get_validation_set()
    test_generator = get_test_set()
    
    
    model = build_model()
    
    print(f"Training model for {EPOCHS} epochs...")
    
    with mlflow.start_run():
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            verbose=1
        )
        
        
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
        
       
        model.save(MODEL_SAVE_PATH)
        print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
        
     
        mlflow.keras.log_model(model, "model")
    
    print("\n🎉 Training completed successfully!")
    print("You can now run CNN_testing.py to evaluate the model on the test set.")
    print("Or run the Streamlit UI with: python -m streamlit run ui.py")


if __name__ == "__main__":
    main()
