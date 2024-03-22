import os
import pandas as pd
import tensorflow as tf
from src.visualize_data import visualize_labels_distribution


def load_data(directory):
    filepath = []
    label = []

    folders = os.listdir(directory)

    for folder in folders:
        f_path = os.path.join(directory, folder)
        imgs = os.listdir(f_path)
        for img in imgs:
            img_path = os.path.join(f_path, img)
            filepath.append(img_path)
            label.append(folder)

    #Concat data paths with labels
    file_path_series = pd.Series(filepath, name= 'filepath')
    Label_path_series = pd.Series(label, name = 'label')
    df = pd.concat([file_path_series, Label_path_series], axis = 1)
    return df


def create_datasets_with_augmentation(data_dir, image_size=(256, 256), batch_size=32):
    # Definir las transformaciones de aumento de datos
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.2)
    ])
    
    # Nota: El aumento de datos se aplica solo al conjunto de entrenamiento
    print('Training Images:')
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/train',
        validation_split=0.1,
        subset='training',
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    ).map(lambda x, y: (data_augmentation(x, training=True), y))

    print('Validation Images:')
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/train',
        validation_split=0.1,
        subset='validation',
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    print('Testing Images:')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/test',
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    # Normalización de los valores de píxeles a [0, 1]
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, validation_ds, test_ds


def extract_labels(dataset):
    labels = []
    for _, y in dataset:
        labels.extend(y.numpy())
    return labels


def create(path, train_dir, test_dir):
    # Load the training and testing data
    train_data = load_data(train_dir)
    test_data = load_data(test_dir)

    # Shape
    print(f"The shape of The Train data is: {train_data.shape}")
    print(f"The shape of The Test data is: {test_data.shape}")

    # Create datasets
    train_ds, validation_ds, test_ds = create_datasets_with_augmentation(path, image_size=(256, 256), batch_size=16)

    # Shape of the dataset
    for image_batch, labels_batch in train_ds:
        print("Shape of X_train: ", image_batch.shape)
        print("Shape of y_train: ", labels_batch.shape)
        break

    # Visualize the distribution of the labels
    visualize_labels_distribution(extract_labels(train_ds), title='Distribution of Categories (train)')

    return train_ds, validation_ds, test_ds