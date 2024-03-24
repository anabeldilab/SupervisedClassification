from src.create_datasets import create_datasets, load_image_path, transform_metrics
from models.xception_94 import xception_94, xception_94_cv
from models.vgg16 import vgg_16, vgg_16_cv
from models.resnet_50 import resnet_50, resnet_50_cv
from models.svm import svm_classifier as svm
from models.knn import knn_classifier as knn
from models.ann_classifier import ann_classifier, ann_classifier_cv
from models.random_forest import random_forest_classifier as random_forest
from src.visualize_data import plot_training_history, save_model_results, save_cv_results, save_mean_metrics, cv_boxplot, plot_learning_curve
from src.hog import HOG
from src.lbp import LBP
import pandas as pd
import numpy as np
from src.hypothesis_testing import hypothesis_testing


# Load the dataset
path = 'data'
train_dir = 'data/train'
test_dir = 'data/test'

# Load the dataset
train_data = load_image_path(train_dir)
test_data = load_image_path(test_dir)

# Shape
print(f"The shape of The Train data is: {train_data.shape}")
print(f"The shape of The Test data is: {test_data.shape}")

# Feature extraction
HOG_train_data_svm = HOG(train_data, orientations=9, pixels_per_cell=(16, 16))

LBP_train_data_svm = LBP(train_data, radius=5, points=8)

HOG_train_data_knn = HOG(train_data, orientations=9, pixels_per_cell=(6, 6))

LBP_train_data_knn = LBP(train_data, radius=5, points=9)

HOG_train_data_rf = HOG(train_data, orientations=5, pixels_per_cell=(13, 13))

LBP_train_data_rf = LBP(train_data, radius=5, points=18)
"""
##########  HOG + ANN  ##########
# Build and test the model
print("############### HOG + ANN ###############")
history, test_metrics = ann_classifier(HOG_train_data_knn, test_data)

##########  CNN  ##########
train_ds, validation_ds, test_ds = create_datasets(path, image_size=(256, 256), batch_size=16)
columns = ['Model', 'Exactitud', 'Sensibilidad', 'Especificidad', 'Precisión', 'F1-Score', 'AUC']

# Train the model xception
xception_history, val_results, test_metrics = xception_94(train_ds, validation_ds, test_ds)
save_model_results(val_results, test_metrics, columns=columns, file_path='results/model_results/results_cnn.csv', include_header=True)
plot_training_history(xception_history, 'Xception')

# Train the model vgg16
train_ds, validation_ds, test_ds = create_datasets(path, image_size=(256, 256), batch_size=16) # Batch size 16 para VGG16 y ResNet50
vgg_16_history, val_results, test_metrics = vgg_16(train_ds, validation_ds, test_ds)
save_model_results(val_results, test_metrics, columns=columns, file_path='results/model_results/results_cnn.csv', include_header=False)
plot_training_history(vgg_16_history, 'VGG16')

# Train the model resnet_50
resnet_50_history, val_results, test_metrics = resnet_50(train_ds, validation_ds, test_ds)
save_model_results(val_results, test_metrics, columns=columns, file_path='results/model_results/results_cnn.csv', include_header=False)
plot_training_history(resnet_50_history, 'ResNet50') 
"""
 ##########  CNN cross validation ##########

##########  HOG + ANN  ##########
print("############### HOG + ANN ###############")
estimator, ann_histories, hog_ann_cv_metrics = ann_classifier_cv(HOG_train_data_svm, epochs=20, folds=5)
save_cv_results(hog_ann_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=True)
for history in ann_histories:
    plot_training_history(history, 'HOG+ANN-CV')
save_mean_metrics(hog_ann_cv_metrics, include_header=True)
#plot_learning_curve(estimator, 'HOG+ANN', HOG_train_data_svm, cv=5, n_jobs=-1, scoring='f1')


##########  Xception  ##########
xception_94_histories, xception_cv_metrics = xception_94_cv(train_data, epochs=20, folds=5)
save_cv_results(xception_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
for history in xception_94_histories:
    plot_training_history(history, 'Xception-CV')
save_mean_metrics(xception_cv_metrics, include_header=False)


##########  VGG16  ##########
vgg_16_histories, vgg_cv_metrics = vgg_16_cv(train_data, epochs=20, folds=5)
save_cv_results(vgg_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
for history in vgg_16_histories:
    plot_training_history(history, 'VGG16-CV')
save_mean_metrics(vgg_cv_metrics, include_header=False)


##########  ResNet50  ##########
resnet_50_histories, resnet_cv_metrics = resnet_50_cv(train_data, epochs=20, folds=5)
save_cv_results(resnet_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
for history in resnet_50_histories:
    plot_training_history(history, 'ResNet50-CV')
save_mean_metrics(resnet_cv_metrics, include_header=False)


##########  HOG + SVM  ##########
# Build and test the model
print("############### HOG + SVM ###############")
estimator, cross_validation, hog_svm_cv_metrics = svm(HOG_train_data_svm, C=79.15, gamma=0.0244)
save_cv_results(hog_svm_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
hog_svm_cv_mean_metrics = save_mean_metrics(hog_svm_cv_metrics, include_header=False)
plot_learning_curve(estimator, 'HOG+SVM', HOG_train_data_svm, cv=cross_validation, n_jobs=-1, scoring='f1')
hog_svm_cv_metrics = transform_metrics(hog_svm_cv_metrics)


##########  LBP + SVM  ##########
# Build and test the model
print("############### LBP + SVM ###############")
estimator, cross_validation, lbp_svm_cv_metrics = svm(LBP_train_data_svm, C=41.3, gamma=1)
save_cv_results(lbp_svm_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
lbp_svm_cv_mean_metrics = save_mean_metrics(lbp_svm_cv_metrics)
plot_learning_curve(estimator, 'LBP+SVM', LBP_train_data_svm, cv=cross_validation, n_jobs=-1, scoring='f1')
lbp_svm_cv_metrics = transform_metrics(lbp_svm_cv_metrics)

##########  HOG + KNN  ##########

# Build and test the model
print("############### HOG + KNN ###############")
estimator, cross_validation, hog_knn_cv_metrics = knn(HOG_train_data_knn, n_neighbors=4)
save_cv_results(hog_knn_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
hog_knn_cv_mean_metrics = save_mean_metrics(hog_knn_cv_metrics)
plot_learning_curve(estimator, 'HOG+KNN', HOG_train_data_knn, cv=cross_validation, n_jobs=-1, scoring='f1')
hog_knn_cv_metrics = transform_metrics(hog_knn_cv_metrics)

##########  LBP + KNN  ##########

# Build and test the model
print("############### LBP + KNN ###############")
estimator, cross_validation, lbp_knn_cv_metrics = knn(LBP_train_data_knn, n_neighbors=5)
save_cv_results(lbp_knn_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
lbp_knn_cv_mean_metrics = save_mean_metrics(lbp_knn_cv_metrics)
plot_learning_curve(estimator, 'LBP+KNN', LBP_train_data_knn, cv=cross_validation, n_jobs=-1, scoring='f1')
lbp_knn_cv_metrics = transform_metrics(lbp_knn_cv_metrics)


##########  HOG + Random Forest  ##########

# Build and test the model
print("############### HOG + Random Forest ###############")
estimator, cross_validation, hog_rf_cv_metrics = random_forest(HOG_train_data_rf, n_estimators=10, max_depth=28)
save_cv_results(hog_rf_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
hog_rf_cv_mean_metrics = save_mean_metrics(hog_rf_cv_metrics)
plot_learning_curve(estimator, 'HOG+Random Forest', HOG_train_data_rf, cv=cross_validation, n_jobs=-1, scoring='f1')
hog_rf_cv_metrics = transform_metrics(hog_rf_cv_metrics)

##########  LBP + Random Forest  ##########

# Build and test the model
print("############### LBP + Random Forest ###############")
estimator, cross_validation, lbp_rf_cv_metrics = random_forest(LBP_train_data_rf, n_estimators=10, max_depth=30)     
save_cv_results(lbp_rf_cv_metrics, file_path='results/model_results/results_cv.csv', include_header=False)
lbp_cv_mean_metrics = save_mean_metrics(lbp_rf_cv_metrics)
plot_learning_curve(estimator, 'LBP+Random Forest', LBP_train_data_rf, cv=cross_validation, n_jobs=-1, scoring='f1')
lbp_rf_cv_metrics = transform_metrics(lbp_rf_cv_metrics)

##########  Boxplot  ##########
models = ['HOG + ANN', 'Xception', 'VGG16', 'ResNet50', 'HOG + SVM', 'LBP + SVM', 'HOG + KNN', 'LBP + KNN', 'HOG + Random Forest', 'LBP + Random Forest']
scores = [hog_ann_cv_metrics['F1-Score'], xception_cv_metrics['F1-Score'], vgg_cv_metrics['F1-Score'], resnet_cv_metrics['F1-Score'], hog_svm_cv_metrics['F1-Score'], lbp_svm_cv_metrics['F1-Score'], hog_knn_cv_metrics['F1-Score'], lbp_knn_cv_metrics['F1-Score'], hog_rf_cv_metrics['F1-Score'], lbp_rf_cv_metrics['F1-Score']]
cv_boxplot(models, scores)


##########  Hypothesis Testing  ##########
scores_models = [
    ('HOG+ANN', hog_ann_cv_metrics['F1-Score']),
    ('Xception', xception_cv_metrics['F1-Score']),
    ('VGG16', vgg_cv_metrics['F1-Score']),
    ('ResNet50', resnet_cv_metrics['F1-Score']),
    ('HOG+SVM', hog_svm_cv_metrics['F1-Score']),
    ('LBP+SVM', lbp_svm_cv_metrics['F1-Score']),
    ('HOG+KNN', hog_knn_cv_metrics['F1-Score']),
    ('LBP+KNN', lbp_knn_cv_metrics['F1-Score']),
    ('HOG+RF', hog_rf_cv_metrics['F1-Score']),
    ('LBP+RF', lbp_rf_cv_metrics['F1-Score'])
]

hypothesis_testing(scores_models, cv=5) 