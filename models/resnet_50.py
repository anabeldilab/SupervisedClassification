from keras import Sequential
from keras.layers import *
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC, FalsePositives, TrueNegatives
import tensorflow.keras.backend as K
import numpy as np
from src.cnn_cv import train_and_evaluate_model_cv
import warnings
warnings.filterwarnings("ignore")

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2*((precision*recall)/(precision+recall+K.epsilon()))


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

    # Compile the model getting accuracy, sensibility, specificity, precision, f1 and auc
    model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall(), F1Score(), FalsePositives(), TrueNegatives()])
    return model, "ResNet50"

############################################################################################################

def train_and_evaluate_model(model, train_ds, validation_ds, epochs=20):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=validation_ds,
        callbacks=[early_stopping]
    )
    # Evaluate the model
    val_metrics = model.evaluate(validation_ds)

    # Calculate specificity
    tn = val_metrics[6]
    fp = val_metrics[5]
    specificity = tn / (tn + fp)

    metrics = {
        'Model': 'ResNet50 - Validation',
        'Exactitud': val_metrics[1],
        'Sensibilidad': val_metrics[3],
        'Especificidad': specificity,
        'Precisión': val_metrics[4],
        'F1-Score': val_metrics[5],
        'AUC': val_metrics[2]
    }

    return history, metrics

############################################################################################################
# MAIN CODE
############################################################################################################

def resnet_50(train_ds, validation_ds, test_ds):
    model = build_model(input_shape=(256, 256, 3))
    model.summary()

    history, val_metrics = train_and_evaluate_model(model, train_ds, validation_ds, epochs=20)

    test_loss, test_accuracy, test_auc, test_precision, test_recall, test_f1, tn, fp = model.evaluate(test_ds)

    specificity = tn / (tn + fp)

    test_metrics = {
        'Model': 'ResNet50 - Test',
        'Exactitud': test_accuracy,
        'Sensibilidad': test_recall,
        'Especificidad': specificity,
        'Precisión': test_precision,
        'F1-Score': test_f1,
        'AUC': test_auc
    }

    return history, val_metrics, test_metrics


################################# Cross Validation #################################

def resnet_50_cv(train_data, epochs=20, folds=5):
    def model_builder():
        return build_model(input_shape=(256, 256, 3))
    
    histories, cv_metrics = train_and_evaluate_model_cv(model_builder, train_data, folds=folds, epochs=epochs)

    return histories, cv_metrics