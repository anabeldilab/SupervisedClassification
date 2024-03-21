import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def load_data(directory):
    filepath =[]
    label = []

    folds = os.listdir(directory)

    for fold in folds:
        f_path = os.path.join(directory , fold)
        imgs = os.listdir(f_path)
        for img in imgs:
            img_path = os.path.join(f_path , img)
            filepath.append(img_path)
            label.append(fold)

    #Concat data paths with labels
    file_path_series = pd.Series(filepath , name= 'filepath')
    Label_path_series = pd.Series(label , name = 'label')
    df_train = pd.concat([file_path_series ,Label_path_series ] , axis = 1)
    return df_train

def create_datasets(data_dir, image_size=(256,256), batch_size=32):
    print('Training Images:')
    # Training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/train',
        validation_split=0.1,
        subset='training',
        seed=123,
        image_size=image_size,
        batch_size=batch_size)

    # Validation dataset
    print('Validation Images:')
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/train',
        validation_split=0.1,
        subset='validation',
        seed=123,
        image_size=image_size,
        batch_size=batch_size)

    print('Testing Images:')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + '/test',
        seed=123,
        image_size=image_size,
        batch_size=batch_size)
    
    return train_ds, validation_ds, test_ds

def create(path, train_dir, test_dir):
    # Load the training and testing data
    train_data = load_data(train_dir)
    test_data = load_data(test_dir)

    # Shape
    print(f"The shape of The Train data is: {train_data.shape}")
    print(f"The shape of The Test data is: {test_data.shape}")

    # Create datasets
    train_ds, validation_ds, test_ds = create_datasets(path, image_size=(256, 256), batch_size=32)

    # Extract labels
    train_labels = train_ds.class_names
    test_labels = test_ds.class_names
    validation_labels = validation_ds.class_names

    # Shape of the dataset
    for image_batch, labels_batch in train_ds:
        print("Shape of X_train: ", image_batch.shape)
        print("Shape of y_train: ", labels_batch.shape)
        break

    # Normalizing Pixel Values
    # Train Data
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    # Val Data
    validation_ds = validation_ds.map(lambda x, y: (x / 255.0, y))
    # Test Data
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

    return train_ds, validation_ds, test_ds