import tensorflow as tf
import keras

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = keras.Sequential([
    keras.layers.Rescaling(1./255),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomContrast(0.2)
])

training_set = keras.preprocessing.image_dataset_from_directory(
    'dataset/training_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

training_set = training_set.map(lambda x, y: (train_datagen(x, training=True), y))

# Preprocessing the Test set
test_set = keras.preprocessing.image_dataset_from_directory(
    'dataset/test_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

test_set = test_set.map(lambda x, y: (x / 255.0, y))

# Part 2 - Building the CNN
# Initialising the CNN
cnn = keras.models.Sequential()
cnn.add(keras.layers.Input(shape=(64, 64, 3)))

# Step 1 - Convolution
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))

# Step 2 - Pooling
cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# Adding a second convolutional layer
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# Step 3 - Flattening
cnn.add(keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(keras.layers.Dense(units=128, activation="relu"))

# Step 5 - Output Layer
cnn.add(keras.layers.Dense(units=1, activation="sigmoid"))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
