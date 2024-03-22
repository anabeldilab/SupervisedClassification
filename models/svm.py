# Importar las bibliotecas necesarias
from sklearn import svm, metrics
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

def get_metrics(val_labels, predicted_labels):
    accuracy = metrics.accuracy_score(val_labels, predicted_labels)
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

def build_model(dataframe):
    # Divide dataframe into features and labels
    data = np.array(dataframe['features'].tolist())
    labels = np.array(dataframe['label'])

    cv_metrics = []

    cross_validation = StratifiedKFold(n_splits=5)

    for fold, (train_index, val_index) in enumerate(cross_validation.split(data, labels)):
        training_data, val_data = data[train_index], data[val_index]
        training_labels, val_labels = labels[train_index], labels[val_index]

        # Crear y entrenar el modelo SVM
        model = svm.SVC(gamma='scale', probability=True)  # Asegúrate de activar probability=True para ROC AUC
        model.fit(training_data, training_labels)

        # Evaluar el modelo
        predicted_labels = model.predict(val_data)
        accuracy, sensibility, specificity, precision, f1 = get_metrics(val_labels, predicted_labels)
        auc = get_auc(model, val_data, val_labels)

        metrics = {
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

    return model, cv_metrics


def evaluate_model(model, test_data):
    # Divide dataframe into features and labels
    test_features = np.array(test_data['features'].tolist())
    test_labels = np.array(test_data['label'])

    test_metrics = []

    # Predecir etiquetas para el conjunto de prueba
    predicted_labels = model.predict(test_features)

    # Calcular y mostrar métricas de rendimiento para el conjunto de prueba
    accuracy, sensibility, specificity, precision, f1 = get_metrics(test_labels, predicted_labels)
    auc = get_auc(model, test_features, test_labels)

    metrics = {
        'Fold': 'Test',
        'Exactitud': accuracy,
        'Sensibilidad': sensibility,
        'Especificidad': specificity,
        'Precisión': precision,
        'F1-Score': f1,
        'AUC': auc
    }

    print(f"Test - Accuracy: {accuracy}, Sensibility: {sensibility}, Specificity: {specificity}, Precision: {precision}, F1-Score: {f1}, AUC: {auc}")
    test_metrics.append(metrics)

    return test_metrics

def svm_classifier(train_data, test_data):
    # Construir el modelo SVM
    model, cv_metrics = build_model(train_data)

    # Evaluar el modelo en el conjunto de prueba
    test_metrics = evaluate_model(model, test_data)

    return cv_metrics, test_metrics