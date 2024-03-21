from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import os

# Cargar el modelo pre-entrenado de VGG16
model = VGG16(weights='imagenet', include_top=False)

# Función para extraer características de las imágenes
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Extraer características para todas las imágenes
features = []
labels = []

for img_path in os.listdir('raw/train/PNEUMONIA'):
    features.append(extract_features('raw/train/PNEUMONIA/' + img_path))
    labels.append(1)

for img_path in os.listdir('raw/train/NORMAL'):
    features.append(extract_features('raw/train/NORMAL/' + img_path))
    labels.append(0)

# Entrenar el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(features, labels)

# Ahora puedes usar `knn.predict()` para clasificar nuevas imágenes
# Supongamos que tienes un conjunto de datos de prueba en una carpeta separada
test_features = []
test_labels = []

for img_path in os.listdir('raw/test/PNEUMONIA'):
    test_features.append(extract_features('raw/test/PNEUMONIA/' + img_path))
    test_labels.append(1)

for img_path in os.listdir('raw/test/NORMAL'):
    test_features.append(extract_features('raw/test/NORMAL/' + img_path))
    test_labels.append(0)

# Ahora puedes usar `knn.predict()` para hacer predicciones en tus datos de prueba
predictions = knn.predict(test_features)

# Y luego comparar estas predicciones con las etiquetas reales
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_labels, predictions)

print("La precisión del modelo en el conjunto de prueba es: ", accuracy)

# Crear un diccionario con los hiperparámetros y los resultados
results = {
    'n_neighbors': [knn.n_neighbors],
    'weights': [knn.weights],
    'algorithm': [knn.algorithm],
    'leaf_size': [knn.leaf_size],
    'p': [knn.p],
    'metric': [knn.metric],
    'accuracy': [accuracy]
}

# Convertir el diccionario a un DataFrame
df = pd.DataFrame(results)

# Guardar el DataFrame en un archivo .csv
df.to_csv('model_hyperparameters_and_results.csv', index=False)

