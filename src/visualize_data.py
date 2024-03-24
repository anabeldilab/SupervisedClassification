import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
import pandas as pd

def visualize_labels_distribution(labels, title='Distribution of Categories'):
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
    palette = sns.color_palette("viridis")
    sns.set_palette(palette)
    axs[0].pie(counts, labels=unique, autopct='%1.1f%%', startangle=140)
    axs[0].set_title(title)
    sns.barplot(x=list(unique), y=list(counts), ax=axs[1], palette="viridis")
    axs[1].set_title('Count of Categories')
    plt.tight_layout()
    plt.show()


def visualize_sample_images(path, num_images=5):
    image_filenames = os.listdir(path)
    num_images = min(num_images, len(image_filenames))
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3), facecolor='white')
    for i, image_filename in enumerate(image_filenames[:num_images]):
        image_path = os.path.join(path, image_filename)
        image = mpimg.imread(image_path)
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)
    plt.tight_layout()
    plt.show()

def plot_training_history(history, model_name='model'):
    best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    axs[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    axs[0].scatter(best_epoch - 1, history.history['val_accuracy'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title(f'Training and Validation Accuracy for {model_name}')
    axs[0].legend()
    axs[1].plot(history.history['loss'], label='Training Loss', color='blue')
    axs[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axs[1].scatter(best_epoch - 1, history.history['val_loss'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training and Validation Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f'results/model_results/graphs/training_history_{model_name}.png')
    plt.close()



def plot_validation_curve(estimator, dataset, model_name, param_name, param_range, cv=None, scoring='f1'):
    """
    Genera una curva de validación para un estimador sklearn utilizando F1-score como métrica.
    
    Parámetros:
        estimator: objeto estimador de sklearn - el modelo a evaluar.
        dataset: diccionario - contiene las características y etiquetas de los datos.
        model_name: str, el nombre del modelo.
        param_name: str, el nombre del parámetro a variar.
        param_range: array-like, los valores del parámetro a probar.
        cv: int, cross-validation generator or an iterable, opcional (default=None)
            Especifica la estrategia de división de cross-validation.
        scoring: str, callable, list/tuple o dict, opcional (default='f1')
            Estrategia para evaluar el rendimiento del modelo en el conjunto de datos de prueba.
    """
    X = np.array(dataset['features'])
    y = np.array(dataset['label'])

    # Encode the labels
    y = np.array([1 if label == 'PNEUMONIA' else 0 for label in y])
    
    # Calcular las puntuaciones de entrenamiento y validación
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    
    # Calcular las medias y desviaciones estándar
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Generar la gráfica
    plt.title(f"Curva de Validación utilizando {scoring} - {model_name}")
    plt.xlabel(param_name)
    plt.ylabel("Puntaje")
    plt.ylim(0.0, 1.1)
    plt.xticks(param_range)
    plt.plot(param_range, train_scores_mean, label="Puntaje en entrenamiento", color="darkorange", lw=2)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color="darkorange", alpha=0.2)
    plt.plot(param_range, test_scores_mean, label="Puntaje en validación cruzada", color="navy", lw=2)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color="navy", alpha=0.2)
    plt.legend(loc="best")
    plt.savefig(f'results/model_results/graphs/validation_curve_{model_name}.png')
    plt.close()


# Define a function to plot images with their true and predicted labels
def plot_images_with_predictions(model, dataset, class_labels, num_images=40, num_images_per_row=5):
    # Generate predictions for a set number of images
    predictions = model.predict(dataset)

    # Shuffle the dataset
    dataset_shuffled = dataset.shuffle(buffer_size=len(dataset))

    plt.figure(figsize=(15, 10))
    for i, (images, labels) in enumerate(dataset_shuffled.take(num_images)):
        # Convert tensor to NumPy array
        images = images.numpy()

        # Iterate over each image in the batch
        for j in range(len(images)):
            if i * num_images_per_row + j < num_images:  # Check if the total number of images exceeds the desired count
                predicted_class = class_labels[np.argmax(predictions[i * num_images_per_row + j])]
                true_class = class_labels[np.argmax(labels[j])]

                plt.subplot(num_images // num_images_per_row + 1, num_images_per_row, i * num_images_per_row + j + 1)
                plt.imshow(images[j].astype("uint8"))
                plt.title(f'True: {true_class}\nPredicted: {predicted_class}')
                plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/model_results/graphs/images_with_predictions.png')
    plt.close()


def plot_learning_curve(estimator, title, dataset, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1'):
    """
    Genera y muestra un gráfico de las curvas de aprendizaje para un estimador, sin especificar los límites de y.
    
    Parámetros:
        estimator: objeto de tipo estimador de sklearn, el modelo a evaluar.
        title: str, título para la gráfica.
        X: array-like, las características del conjunto de datos.
        y: array-like, las etiquetas del conjunto de datos.
        cv: int, cross-validation generator or an iterable, especifica la estrategia de división de cross-validation.
        n_jobs: int, número de trabajos para computar en paralelo.
        train_sizes: array-like, tamaños de los conjuntos de entrenamiento a usar.
        scoring: str, métrica de puntuación a utilizar.
    """
    X = np.array(dataset['features'])
    y = np.array(dataset['label'])

    # Encode the labels
    y = np.array([1 if label == 'PNEUMONIA' else 0 for label in y])

    plt.figure(figsize=(10, 6))
    plt.title(f"Curvas de Aprendizaje {title}")
    plt.xlabel("Datos de Entrenamiento")
    plt.ylabel(scoring.capitalize())
    plt.grid(True)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Gráfico de las curvas de aprendizaje
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color="r", alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color="g", alpha=0.1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación Cruzada")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'results/model_results/graphs/learning_curve_{title}.png')
    plt.close()



def plot_confusion_matrix(y_true, y_pred, class_labels):
    confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('results/model_results/graphs/confusion_matrix.png')
    plt.close()


def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('results/model_results/graphs/roc_curve.png')
    plt.close()



""" def simulate_learning_curve(model_builder, model_name, get_dataset_fn, train_paths, train_labels, cv, train_sizes=np.linspace(0.1, 1.0, 5)):
    scores = []  # Puntuaciones para cada tamaño de conjunto de entrenamiento
    
    for train_size in train_sizes:
        size_scores = []  # Puntuaciones para este tamaño de conjunto de entrenamiento
        
        for train_index, val_index in cv.split(train_paths, train_labels):
            # Crear subconjuntos de entrenamiento y validación para este fold
            train_ds = get_dataset_fn(train_paths[train_index], train_labels[train_index])
            val_ds = get_dataset_fn(train_paths[val_index], train_labels[val_index])
            
            # Construye y entrena el modelo
            model = model_builder()
            history = model.fit(train_ds, validation_data=val_ds, epochs=20, verbose=0)  # Ajusta según sea necesario
            
            # Evalúa el modelo
            score = model.evaluate(val_ds, verbose=0)
            size_scores.append(score[1])  # Asume que el score de interés es el segundo elemento de la evaluación
            
        # Calcula el promedio de las puntuaciones para este tamaño de conjunto de entrenamiento
        scores.append(np.mean(size_scores))
    
    # Grafica la curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, scores, 'o-', color="r", label="F1-Score")
    plt.title(f'Curva de Aprendizaje - {model_name}')
    plt.xlabel('Fracción del Tamaño de Entrenamiento')
    plt.ylabel('F1-Score')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/model_results/graphs/learning_curve_{model_name}.png')
    plt.close() """





def cv_boxplot(models, scores):  # models = ['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5'] # scores = [score1, score2, score3, score4, score5]
    fig7, ax = plt.subplots()
    ax.set_title('Comparación de Rendimiento de Modelos')
    ax.boxplot(scores, labels=models)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel('Puntuación')
    plt.tight_layout()
    plt.savefig('results/model_results/graphs/cv_boxplot.png')
    plt.close()


def save_model_results(val_results, test_results, columns=['Model', 'Exactitud', 'Sensibilidad', 'Especificidad', 'Precisión', 'F1-Score', 'AUC'], file_path='results/model_results/results.csv', include_header=True):
    # Crear DataFrames para los resultados de validación y prueba
    val_results_df = pd.DataFrame([val_results], columns=columns)
    test_results_df = pd.DataFrame([test_results], columns=columns)

    results = pd.concat([val_results_df, test_results_df])
    
    # Guardar en CSV
    results.to_csv(file_path, mode='a', header=include_header, index=False)


def save_cv_results(cv_results, file_path='results/model_results/results_cv.csv', include_header=True):
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(file_path, mode='a', header=include_header, index=False)

# Metric mean 
def save_mean_metrics(results, columns=['Model', 'Fold', 'Exactitud', 'Sensibilidad', 'Especificidad', 'Precisión', 'F1-Score', 'AUC'], include_header=True):
    # Convertir la lista de diccionarios a un DataFrame
    results_df = pd.DataFrame(results)
    print(f"Results: {results_df}")
    
    # Asegurarse de que todas las columnas esperadas estén presentes en results_df
    if not all(col in results_df for col in columns):
        print("Una o más columnas especificadas no se encuentran en el DataFrame.")
        return pd.DataFrame()  # Retorna un DataFrame vacío para manejar el error
    
    # Convertir las columnas numéricas a tipo numérico, excepto 'Model' y 'Fold'
    num_columns = [col for col in columns if col not in ['Model', 'Fold']]
    for col in num_columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
    # Eliminar la columna 'Fold' antes de calcular el promedio, ya que no es necesario promediar 'Fold'
    results_df = results_df.drop(columns='Fold')

    # Calcular el promedio de las métricas por modelo
    mean_results = results_df.groupby('Model').mean().reset_index()

    print(f"Mean results: {mean_results}")
    
    # Guardar los resultados promedio en un archivo CSV
    mean_results.to_csv('results/model_results/mean_results.csv', mode='a', index=False, header=include_header)
    
    return mean_results