import numpy as np
import tensorflow as tf
import cv2
from sklearn import preprocessing
from matplotlib import pyplot

DATA_PATH = "../../data/digits_and_characters_1/"
OWN_DATA_PATH = "../../own_data/"


def load_data(use_premade_dataset=True):
    # Encode string labels to integers
    label_encoder = preprocessing.LabelEncoder()

    train_images = np.zeros(0)
    train_labels = np.zeros(0)
    validation_images = np.zeros(0)
    validation_labels = np.zeros(0)
    test_images = np.zeros(0)
    test_labels = np.zeros(0)

    if use_premade_dataset:
        data = np.load(f'{DATA_PATH}CompleteDataSet_testing_tuples.npy', allow_pickle=True)
        train_images = np.stack(data[:, 0])
        train_labels = np.stack(data[:, 1])

        validation_data = np.load(f'{DATA_PATH}CompleteDataSet_validation_tuples.npy', allow_pickle=True)
        validation_images = np.stack(validation_data[:, 0])
        validation_labels = np.stack(validation_data[:, 1])

        test_data = np.load(f'{DATA_PATH}CompleteDataSet_testing_tuples.npy', allow_pickle=True)
        test_images = np.stack(test_data[:, 0])
        test_labels = np.stack(test_data[:, 1])

    # TODO: Add other datasets
    own_data_labels = np.load(f'{OWN_DATA_PATH}robert_handwritten_dataset_labels.npy', allow_pickle=True)
    own_data_images_gray = np.load(f'{OWN_DATA_PATH}robert_handwritten_dataset_images.npy', allow_pickle=True)

    # Thresholding to have same format as other dataset
    (thresh, own_data_images) = cv2.threshold(own_data_images_gray, 127, 255, cv2.THRESH_BINARY)
    own_data_images = (own_data_images / 255).astype(int)

    # Split data into train/test/validation sets
    own_data_size = own_data_labels.size
    test_split = int(own_data_size * 0.15)
    validation_split = int(own_data_size * 0.1)
    own_test_data_images = own_data_images[0:test_split]
    own_test_data_labels = own_data_labels[0:test_split]
    own_validation_data_images = own_data_images[test_split:validation_split + test_split]
    own_validation_data_labels = own_data_labels[test_split:validation_split + test_split]
    own_training_data_images = own_data_images[validation_split + test_split:]
    own_training_data_labels = own_data_labels[validation_split + test_split:]

    if use_premade_dataset:
        train_images = np.concatenate((train_images, own_training_data_images))
        train_labels = np.concatenate((train_labels, own_training_data_labels))
        test_images = np.concatenate((test_images, own_test_data_images))
        test_labels = np.concatenate((test_labels, own_test_data_labels))
        validation_images = np.concatenate((validation_images, own_validation_data_images))
        validation_labels = np.concatenate((validation_labels, own_validation_data_labels))
    else:
        train_images = own_training_data_images
        train_labels = own_training_data_labels
        test_images = own_test_data_images
        test_labels = own_test_data_labels
        validation_images = own_validation_data_images
        validation_labels = own_validation_data_labels

    label_encoder.fit(train_labels)
    train_classes = label_encoder.transform(train_labels)
    test_classes = label_encoder.transform(test_labels)
    validation_classes = label_encoder.transform(validation_labels)

    return train_images, train_classes, test_images, test_classes, validation_images, validation_classes, label_encoder


# Load data
train_images, train_classes, test_images, test_classes, validation_images, validation_classes, label_encoder = load_data(True)

# Save the encoder
np.save('../../models/test_model/classes.npy', label_encoder.classes_)

# Ensure we only use a fixed amount of memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Set memory limit to something lower than total GPU memory
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
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
# model.summary()

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
model.save('../../models/test_model')
