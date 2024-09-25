import tensorflow as tf

# Preprocessing the Training set
train_datagen = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.2)
])

training_set = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/training_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

training_set = training_set.map(lambda x, y: (train_datagen(x, training=True), y))

# Preprocessing the Test set
test_set = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/test_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

test_set = test_set.map(lambda x, y: (x / 255.0, y))