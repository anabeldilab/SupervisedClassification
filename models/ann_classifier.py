from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.metrics import Precision, Recall, AUC, FalsePositives, TrueNegatives
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import numpy as np
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


def build_model(features):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(features.shape[1],), kernel_regularizer=l2(0.004)),
        Dropout(0.4829),
        Dense(64, activation='relu', kernel_regularizer=l2(0.004)),
        Dropout(0.1561),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adamax(learning_rate=0.000069), loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall(), F1Score(), FalsePositives(), TrueNegatives()])
    return model



def ann_classifier(train_data, test_data):
    train_features = np.array(train_data['features'])
    train_labels = train_data['label']

    # Encode labels
    train_labels = np.array([1 if label == 'PNEUMONIA' else 0 for label in train_labels])

    scaler = MinMaxScaler(feature_range=(0, 1))
    features_train_scaled = scaler.fit_transform(train_features)


    model = build_model(features_train_scaled)

    test_features = np.array(test_data['features'])
    test_labels = test_data['label']

    # Encode labels
    test_labels = np.array([1 if label == 'PNEUMONIA' else 0 for label in test_labels])

    features_test_scaled = scaler.transform(test_features)

    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitorea la pérdida de validación
        min_delta=0.001,  # Cambio mínimo considerado como mejora
        patience=10,  # Número de épocas sin mejora después de las cuales el entrenamiento se detendrá
        verbose=1,
        mode='min',  # Porque estamos monitoreando la pérdida de validación que debe minimizarse
        restore_best_weights=True  # Restaura los pesos del modelo desde la época con el mejor valor de la métrica monitoreada.
    )

    # Entrenamiento del modelo con Early Stopping
    history = model.fit(
        features_train_scaled,
        train_labels,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    test_loss, test_accuracy, *other_metrics = model.evaluate(features_test_scaled, test_labels)
    print(f'Test Accuracy: {test_accuracy}')

    return history, test_accuracy 

################################# Cross Validation #################################

def ann_classifier_cv(train_data, epochs=20, folds=5):
    train_features = np.array(train_data['features'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_train_scaled = scaler.fit_transform(train_features)
    def model_builder():
        return build_model(features_train_scaled)
    
    # Entrenar y evaluar el modelo usando validación cruzada
    histories, cv_metrics = train_and_evaluate_model_cv(model_builder, "ANN", train_data, folds=folds, epochs=epochs)

    estimator = KerasClassifier(build_fn=model_builder, epochs=epochs, batch_size=16, verbose=1)
    
    return estimator, histories, cv_metrics


def get_dataset(features, labels, batch_size=16):
    labels = np.array([1 if label == 'PNEUMONIA' else 0 for label in labels])
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def train_and_evaluate_model_cv(model_builder, model_name, dataframe, folds=5, epochs=20):
    data = np.array(dataframe['features'])
    labels = np.array(dataframe['label'])    
    
    # Inicializa el StratifiedKFold
    cross_validation = StratifiedKFold(n_splits=folds, shuffle=True)

    # Historiales de entrenamiento y métricas de validación para cada fold
    histories = []
    cv_metrics = []

    for fold, (train_index, val_index) in enumerate(cross_validation.split(data, labels)):
        print(f"Entrenando fold {fold + 1}/{folds}")
        training_data, val_data = data[train_index], data[val_index]
        training_labels, val_labels = labels[train_index], labels[val_index]
        
        # Crea datasets de entrenamiento y validación para el fold actual
        train_ds = get_dataset(training_data, training_labels)
        val_ds = get_dataset(val_data, val_labels)

        # Construye el modelo (esta función debe construir, compilar y devolver tu modelo)
        model = model_builder()

        # Entrenamiento
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)
        histories.append(history)

        # Evaluación
        val_metrics = model.evaluate(val_ds, verbose=1)

        # Calculate specificity
        tn = val_metrics[6]
        fp = val_metrics[5]
        specificity = tn / (tn + fp)
        print(f'Specificity: {specificity}')

        metrics = {
            'Model': model_name + ' - Validation',
            'Fold': fold + 1,
            'Exactitud': val_metrics[1],
            'Sensibilidad': val_metrics[3],
            'Especificidad': specificity,
            'Precisión': val_metrics[4],
            'F1-Score': val_metrics[5],
            'AUC': val_metrics[2]
        }

        cv_metrics.append(metrics)

    return histories, cv_metrics


def get_metrics(val_labels, predicted_labels):
    accuracy = accuracy_score(val_labels, predicted_labels)
    cm = confusion_matrix(val_labels, predicted_labels)
    
    # Asumiendo 'PNEUMONIA' como la etiqueta positiva
    sensibility = recall_score(val_labels, predicted_labels, pos_label='PNEUMONIA')
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    precision = precision_score(val_labels, predicted_labels, pos_label='PNEUMONIA')
    f1 = f1_score(val_labels, predicted_labels, pos_label='PNEUMONIA')

    return accuracy, sensibility, specificity, precision, f1


def get_auc(model, val_data, val_labels):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(val_data)
    else:
        scores = model.predict_proba(val_data)[:, 1]
    auc = roc_auc_score(val_labels, scores)
    return auc