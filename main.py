from src.create_datasets import create, load_image_path
from models.xception_94 import xception_94
from models.vgg16 import vgg_16
from models.resnet_50 import resnet_50
from models.svm import svm_classifier as svm
from models.knn import knn_classifier as KNN
from models.random_forest import random_forest_classifier as random_forest
from src.visualize_data import plot_training_history
from src.hog import HOG
from src.lbp import LBP


# Load the dataset
path = 'data'
train_dir = 'data/train'
test_dir = 'data/test'

# Load the dataset
train_data = load_image_path(train_dir)
test_data = load_image_path(test_dir)

# Feature extraction
HOG_train_features = HOG(train_data['filepath'])
HOG_test_features = HOG(test_data['filepath'])

LBP_train_features = LBP(train_data['filepath'])
LBP_test_features = LBP(test_data['filepath'])

# Unify features and labels
HOG_train_data = {
    'features': HOG_train_features,
    'label': train_data['label']
}

HOG_test_data = {
    'features': HOG_test_features,
    'label': test_data['label']
}

LBP_train_data = {
    'features': LBP_train_features,
    'label': train_data['label']
}

LBP_test_data = {
    'features': LBP_test_features,
    'label': test_data['label']
}  

##########  CNN  ##########

""" train_ds, validation_ds, test_ds = create(path, train_dir, test_dir)

# Train the model xception
xception_history = xception_94(train_ds, validation_ds, test_ds)

# Train the model vgg16
vgg_16_history = vgg_16(train_ds, validation_ds, test_ds)

# Train the model resnet_50
resnet_50_history = resnet_50(train_ds, validation_ds, test_ds)

# Plot the training history
plot_training_history(xception_history)
plot_training_history(vgg_16_history)
plot_training_history(resnet_50_history) """

##########  HOG + SVM  ##########

# Build and test the model
print("############### HOG + SVM ###############")
cv_metrics, test_metrics = svm(HOG_train_data, HOG_test_data)

##########  LBP + SVM  ##########

# Build and test the model
print("############### LBP + SVM ###############")
cv_metrics, test_metrics = svm(LBP_train_data, LBP_test_data)

##########  HOG + KNN  ##########

# Build and test the model
print("############### HOG + KNN ###############")
cv_metrics, test_metrics = KNN(HOG_train_data, HOG_test_data)

##########  LBP + KNN  ##########

# Build and test the model
print("############### LBP + KNN ###############")
cv_metrics, test_metrics = KNN(LBP_train_data, LBP_test_data)

##########  HOG + Random Forest  ##########

# Build and test the model
print("############### HOG + Random Forest ###############")
cv_metrics, test_metrics = random_forest(HOG_train_data, HOG_test_data)

##########  LBP + Random Forest  ##########

# Build and test the model
print("############### LBP + Random Forest ###############")
cv_metrics, test_metrics = random_forest(LBP_train_data, LBP_test_data)






