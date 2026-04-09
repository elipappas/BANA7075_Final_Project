import tensorflow as tf
from Image_Preparation import get_training_set, get_validation_set

training_set = get_training_set()
validation_set = get_validation_set()

cnn = tf.keras.models.Sequential()

# First Convolution + Pooling
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(224, 224, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten
cnn.add(tf.keras.layers.Flatten())

# Fully connected layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Dropout to reduce overfitting
cnn.add(tf.keras.layers.Dropout(0.5))

# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = cnn.fit(training_set, validation_data=validation_set, epochs=5)

cnn.save('xray_cnn_model.keras')