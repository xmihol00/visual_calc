import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn import preprocessing
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from const_config import DIGIT_AND_OPERATORS_1_TRAIN
from const_config import DIGIT_AND_OPERATORS_1_VALIDATION
from const_config import DIGIT_AND_OPERATORS_1_TEST
from const_config import RESULTS_PATH
from const_config import MODELS_PATH
from const_config import OWN_DATA_PATH
from const_config import SEED

def load_data(own_data_augmentation=True):
    # Encode string labels to integers
    label_encoder = preprocessing.LabelEncoder()

    train_images = np.zeros(0)
    train_labels = np.zeros(0)
    validation_images = np.zeros(0)
    validation_labels = np.zeros(0)
    test_images = np.zeros(0)
    test_labels = np.zeros(0)

    premade_data = np.load(DIGIT_AND_OPERATORS_1_TRAIN, allow_pickle=True)
    fit_labels = np.stack(premade_data[:, 1])
    indices_left = np.where(fit_labels == "[")
    fit_labels = np.delete(fit_labels, indices_left[0])
    indices_right = np.where(fit_labels == "]")
    fit_labels = np.delete(fit_labels, indices_right[0])

    train_images = np.stack(premade_data[:, 0])
    train_images = np.delete(train_images, indices_left[0], axis=0)
    train_images = np.delete(train_images, indices_right[0], axis=0)
    train_labels = fit_labels

    validation_data = np.load(DIGIT_AND_OPERATORS_1_VALIDATION, allow_pickle=True)
    validation_labels = np.stack(validation_data[:, 1])
    val_indices_left = np.where(validation_labels == "[")
    validation_labels = np.delete(validation_labels, val_indices_left[0])
    val_indices_right = np.where(validation_labels == "]")
    validation_labels = np.delete(validation_labels, val_indices_right[0])
    validation_images = np.stack(validation_data[:, 0])
    validation_images = np.delete(validation_images, val_indices_left[0], axis=0)
    validation_images = np.delete(validation_images, val_indices_right[0], axis=0)

    test_data = np.load(DIGIT_AND_OPERATORS_1_TEST, allow_pickle=True)
    test_labels = np.stack(test_data[:, 1])
    test_indices_left = np.where(test_labels == "[")
    test_labels = np.delete(test_labels, test_indices_left[0])
    test_indices_right = np.where(test_labels == "]")
    test_labels = np.delete(test_labels, test_indices_right[0])
    test_images = np.stack(test_data[:, 0])
    test_images = np.delete(test_images, test_indices_left[0], axis=0)
    test_images = np.delete(test_images, test_indices_right[0], axis=0)

    if own_data_augmentation:
        own_data_labels = np.load(f'{OWN_DATA_PATH}robert_handwritten_dataset_labels.npy', allow_pickle=True)
        own_data_images_gray = np.load(f'{OWN_DATA_PATH}robert_handwritten_dataset_images.npy', allow_pickle=True)

        # Thresholding to have same format as other dataset
        _, own_data_images = cv2.threshold(own_data_images_gray, 127, 255, cv2.THRESH_BINARY)
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

    if own_data_augmentation:
        train_images = np.concatenate((train_images, own_training_data_images))
        train_labels = np.concatenate((train_labels, own_training_data_labels))
        test_images = np.concatenate((test_images, own_test_data_images))
        test_labels = np.concatenate((test_labels, own_test_data_labels))
        validation_images = np.concatenate((validation_images, own_validation_data_images))
        validation_labels = np.concatenate((validation_labels, own_validation_data_labels))

    label_encoder.fit(fit_labels)
    train_classes = label_encoder.transform(train_labels)
    test_classes = label_encoder.transform(test_labels)
    validation_classes = label_encoder.transform(validation_labels)

    return train_images, train_classes, test_images, test_classes, validation_images, validation_classes, label_encoder

if __name__ == "__main__":
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Train the neural network.")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate the neural network.")
    parser.add_argument("-a", "--augmentation", action="store_true", help="Use augmented data set.")
    args = parser.parse_args()

    # Load data
    train_images, train_classes, test_images, test_classes, validation_images, validation_classes, label_encoder = load_data(args.augmentation)
    
    # Save the encoder
    os.makedirs(f'{MODELS_PATH}encoder', exist_ok=True)
    np.save(f'{MODELS_PATH}encoder/classes.npy', label_encoder.classes_)

    # Recognition model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(14, activation='softmax')
    ])

    # Ensure we only use a fixed amount of memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Set memory limit to something lower than total GPU memory
                tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if args.train: # Train the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_images, train_classes,
                            validation_data=(validation_images, validation_classes),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)],
                            epochs=100,
                            verbose=2)

        # Store the trained model
        os.makedirs(f'{MODELS_PATH}classifier_{"own_data_augmentation" if args.augmentation else "no_augmentation"}', exist_ok=True)
        model.save(f'{MODELS_PATH}classifier_{"own_data_augmentation" if args.augmentation else "no_augmentation"}/mser_classifier.h5', save_format="h5")

        # Plot the results of training
        figure, axis = plt.subplots(2, 1)
        figure.set_size_inches(6, 6)
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, hspace=0.2, wspace=0.02)

        axis[0].set_title('Loss')
        axis[0].plot(history.history['loss'], label='train')
        axis[0].plot(history.history['val_loss'], label='validation')
        axis[0].legend()

        axis[1].set_title('Accuracy')
        axis[1].plot(history.history['accuracy'], label='train')
        axis[1].plot(history.history['val_accuracy'], label='validation')
        axis[1].legend()

        plt.savefig(f"{RESULTS_PATH}classifier_training_progress", dpi=400)
        plt.close()

    elif args.evaluate: # Evaluate the model
        model.load_weights(f'{MODELS_PATH}classifier_{"own_data_augmentation" if args.augmentation else "no_augmentation"}').expect_partial()
        labels = label_encoder.classes_
        total_predictions = model.predict(test_images, verbose=2)
        highest_probability = np.argmax(total_predictions, axis=1)
        labels[0] = "/"

        # plot confusion matrix
        figure, axis = plt.subplots(1, 1)
        figure.set_size_inches(10, 8.6)
        plt.subplots_adjust(left=-0.03, bottom=0.07, right=1.05, top=0.96, hspace=0.1, wspace=0.02)
        confusion_matrix = tf.math.confusion_matrix(test_classes, highest_probability).numpy()
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
        cm_display.plot(cmap="Blues", ax=axis)
        plt.savefig(f"{RESULTS_PATH}classifier_confusion_matrix", dpi=400)
        plt.show()
        plt.close()
