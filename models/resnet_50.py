from keras import Sequential
from keras.layers import *
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

def build_model(input_shape=(256, 256, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
    base_model.trainable = False
    model = Sequential([
        base_model,
        BatchNormalization(),
        Dropout(0.45),
        Dense(220, activation='relu'),
        Dropout(0.25),
        Dense(60, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

############################################################################################################

def train_and_evaluate_model(model, train_ds, validation_ds, epochs=20):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=validation_ds,
        callbacks=[early_stopping]
    )
    validation_loss, validation_accuracy = model.evaluate(validation_ds)
    print(f"Validation Loss: {validation_loss}")
    print(f"Validation Accuracy: {validation_accuracy}")
    return history

############################################################################################################
# MAIN CODE
############################################################################################################

def resnet_50(train_ds, validation_ds, test_ds):
    # Build the model
    model = build_model(input_shape=(256, 256, 3))
    model.summary()

    # Train and evaluate the model
    history = train_and_evaluate_model(model, train_ds, validation_ds, epochs=20)

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    return history


