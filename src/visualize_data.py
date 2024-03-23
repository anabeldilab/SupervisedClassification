import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve

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

def plot_training_history(history):
    best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    axs[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    axs[0].scatter(best_epoch - 1, history.history['val_accuracy'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].legend()
    axs[1].plot(history.history['loss'], label='Training Loss', color='blue')
    axs[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axs[1].scatter(best_epoch - 1, history.history['val_loss'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training and Validation Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

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
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_labels):
    confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=class_labels, yticklabels=class_labels)


def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



""" def validation_curve(param_range, train_scores, val_scores, title, x_label, y_label):
    ''' Plot validation curve
    param_range: list of parameter values
    train_scores: list of training scores
    val_scores: list of validation scores
    title: title of the plot
    x_label: label of the x-axis
    y_label: label of the y-axis    
    
    '''
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 7))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0.0, 1.1)
    lw = 2

    plt.semilogx(param_range, train_scores_mean, label='Training score', color='darkorange', lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='darkorange')

    plt.semilogx(param_range, val_scores_mean, label='Cross-validation score', color='navy', lw=lw)
    plt.fill_between(param_range, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.2, color='navy')

    plt.legend(loc='best')
    plt.show() """


def cv_boxplot(models, scores): # models = ['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5'] scores = [score1, score2, score3, score4, score5]
    fig7, ax = plt.subplots()
    ax.set_title('Modelos')
    ax.boxplot(scores,labels=models)