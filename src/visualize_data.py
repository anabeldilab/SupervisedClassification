import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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