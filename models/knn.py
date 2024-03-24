# Importar las bibliotecas necesarias
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

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

def build_model(dataframe, n_neighbors=5):
    # Divide dataframe into features and labels
    data = np.array(dataframe['features'])
    labels = np.array(dataframe['label'])

    cv_metrics = []

    cross_validation = StratifiedKFold(n_splits=5)

    for fold, (train_index, val_index) in enumerate(cross_validation.split(data, labels)):
        training_data, val_data = data[train_index], data[val_index]
        training_labels, val_labels = labels[train_index], labels[val_index]

        # Crear y entrenar el modelo SVM
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        estimator = model.fit(training_data, training_labels)

        # Evaluar el modelo
        predicted_labels = model.predict(val_data)
        accuracy, sensibility, specificity, precision, f1 = get_metrics(val_labels, predicted_labels)
        auc = get_auc(model, val_data, val_labels)

        metrics = {
            'Model': 'KNN - Validation',
            'Fold': fold + 1,
            'Exactitud': accuracy,
            'Sensibilidad': sensibility,
            'Especificidad': specificity,
            'Precisión': precision,
            'F1-Score': f1,
            'AUC': auc
        }

        print(f"Fold {fold + 1} - Accuracy: {accuracy}, Sensibility: {sensibility}, Specificity: {specificity}, Precision: {precision}, F1-Score: {f1}, AUC: {auc}")
        cv_metrics.append(metrics)

    return estimator, cross_validation, cv_metrics


def knn_classifier(train_data, n_neighbors=5):
    # Construir el modelo SVM
    estimator, cross_validation, cv_metrics = build_model(train_data, n_neighbors)

    return estimator, cross_validation, cv_metrics