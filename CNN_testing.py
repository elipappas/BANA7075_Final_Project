import tensorflow as tf
from tensorflow import keras
from Image_Preparation import get_test_set

test_set = get_test_set()

cnn = keras.models.load_model("xray_cnn_model.keras")

test_loss, test_acc = cnn.evaluate(test_set)
print("Test accuracy:", test_acc)