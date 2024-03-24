import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from src.visualize_data import simulate_learning_curve
import matplotlib.pyplot as plt

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0  # Normalizar a [0,1]
    return image, label


def get_dataset(paths, labels, batch_size=16):
    labels = np.array([1 if label == 'PNEUMONIA' else 0 for label in labels])
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def train_and_evaluate_model_cv(model_builder, dataframe, folds=5, epochs=20):
    paths = np.array(dataframe['filepath'])
    labels = np.array(dataframe['label'])       
    
    # Inicializa el StratifiedKFold
    cross_validation = StratifiedKFold(n_splits=folds, shuffle=True)

    # Historiales de entrenamiento y métricas de validación para cada fold
    histories = []
    cv_metrics = []
    scores = []

    for train_size in np.linspace(0.1, 1.0, 5):
        size_scores = []
        for fold, (train_idx, val_idx) in enumerate(cross_validation.split(paths, labels)):
            print(f"Entrenando fold {fold + 1}/{folds}")

            # Crea datasets de entrenamiento y validación para el fold actual
            train_ds = get_dataset(paths[train_idx], labels[train_idx])
            val_ds = get_dataset(paths[val_idx], labels[val_idx])

            # Construye el modelo (esta función debe construir, compilar y devolver tu modelo)
            model, model_name = model_builder()

            # Entrenamiento
            history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)
            histories.append(history)

            # Evaluación
            val_metrics = model.evaluate(val_ds, verbose=1)

            size_scores.append(val_metrics[5])

            # Calculate specificity
            tn = val_metrics[6]
            fp = val_metrics[5]
            specificity = tn / (tn + fp)

            metrics = {
                'Model': model_name + ' - Validation',
                'Exactitud': val_metrics[1],
                'Sensibilidad': val_metrics[3],
                'Especificidad': specificity,
                'Precisión': val_metrics[4],
                'F1-Score': val_metrics[5],
                'AUC': val_metrics[2]
            }

            scores.append(np.mean(size_scores))
            cv_metrics.append(metrics)

    # Grafica la curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0.1, 1.0, 5), scores, 'o-', color="r", label="F1-Score")
    plt.title(f'Curva de Aprendizaje - {model_name}')
    plt.xlabel('Fracción del Tamaño de Entrenamiento')
    plt.ylabel('F1-Score')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/model_results/graphs/learning_curve_{model_name}.png')
    plt.close()

    return histories, cv_metrics


def get_auc(model, val_data, val_labels):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(val_data)
    else:
        scores = model.predict_proba(val_data)[:, 1]
    auc = roc_auc_score(val_labels, scores)

    return auc


def get_metrics(val_labels, predicted_labels):
    accuracy = accuracy_score(val_labels, predicted_labels)
    cm = confusion_matrix(val_labels, predicted_labels)
    
    # Asumiendo 'PNEUMONIA' como la etiqueta positiva
    sensibility = recall_score(val_labels, predicted_labels, pos_label='PNEUMONIA')
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    precision = precision_score(val_labels, predicted_labels, pos_label='PNEUMONIA')
    f1 = f1_score(val_labels, predicted_labels, pos_label='PNEUMONIA')

    return accuracy, sensibility, specificity, precision, f1
