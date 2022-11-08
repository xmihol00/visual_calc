import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from matplotlib import pyplot

DATA_PATH = "../data/digits_and_operators_1/"

# Encode string labels to integers
labelEncoder = preprocessing.LabelEncoder()

# Load data
data = np.load(f'{DATA_PATH}CompleteDataSet_testing_tuples.npy', allow_pickle=True)
train_images = np.stack(data[:, 0])
train_labels = np.stack(data[:, 1])
labelEncoder.fit(train_labels)
train_classes = labelEncoder.transform(train_labels)

validation_data = np.load(f'{DATA_PATH}CompleteDataSet_validation_tuples.npy', allow_pickle=True)
validation_images = np.stack(validation_data[:, 0])
validation_labels = np.stack(validation_data[:, 1])
validation_classes = labelEncoder.transform(validation_labels)

test_data = np.load(f'{DATA_PATH}CompleteDataSet_testing_tuples.npy', allow_pickle=True)
test_images = np.stack(test_data[:, 0])
test_labels = np.stack(test_data[:, 1])
test_classes = labelEncoder.transform(test_labels)

# Save the encoder
np.save('../models/test_model/classes.npy', labelEncoder.classes_)

# Ensure we only use a fixed amount of memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Set memory limit to something lower than total GPU memory
            tf.config.set_logical_device_configuration(gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Recognition model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(16, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the Keras model:
#model.summary()

# Train and evaluate model
history = model.fit(train_images, train_classes,
                    validation_data=(validation_images, validation_classes),
                    epochs=12)
test_loss, test_acc = model.evaluate(test_images, test_classes, verbose=2)

# Plot the results of training
print('Test loss: %.3f, Test acc: %.3f' % (test_loss, test_acc))
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# Store the trained model
model.save('../models/test_model')
