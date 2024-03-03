from os import path, makedirs, listdir
from skimage import io, transform, filters, img_as_ubyte
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size=(128, 128), categories=['NORMAL', 'PNEUMONIA'], data_types=['test', 'train', 'val']):
        self.target_size = target_size
        self.categories = categories
        self.data_types = data_types


    def resize_image(self, image):
        """ Resizes an image to the target size. """
        return transform.resize(image, self.target_size, mode='reflect')


    def apply_filter(self, image, filter_type='gaussian'):
        """ Applies a filter to the image. Only supports 'gaussian' for now."""
        if filter_type == 'gaussian':
            return filters.gaussian(image, sigma=1)
        else:
            raise ValueError("Filter type not supported")


    def process_image(self, image_path):
        """ Processes a single image."""
        image = io.imread(image_path)
        image_resized = self.resize_image(image)
        image_filtered = self.apply_filter(image_resized)
        return image_filtered


    def process_directory_to_file(self, directory_path, output_directory):
        """ Processes all the images in a directory and saves the processed images to an output directory."""
        for filename in listdir(directory_path):
            if filename.endswith('.jpeg'): # Assuming images in JPG format
                print(f"Processing {filename}")
                file_path = build_path(directory_path, filename)
                processed_image = self.process_image(file_path)
                # Convert the processed image to 'uint8'
                processed_image_uint8 = img_as_ubyte(processed_image)
                create_path(output_directory)
                output_image_path = build_path(output_directory, filename)
                # Save the processed image
                io.imsave(output_image_path, processed_image_uint8)


    def process_dataset(self):
        """ Processes the entire dataset."""
        if path.exists(f'data/processed_{self.target_size[0]}_{self.target_size[1]}'):
            print(f"Processed dataset already exists. Skipping processing.")
            return
        for data_type in self.data_types:
            for category in self.categories:
                input_image_path = f'data/raw/{data_type}/{category}/'
                output_image_path = f'data/processed_{self.target_size[0]}_{self.target_size[1]}/{data_type}/{category}/'
                
                print(f"Processing {data_type}/{category} images to {output_image_path}")
                self.process_directory_to_file(input_image_path, output_image_path)


# Auxiliary functions for better code readability

# Create path function if it doesn't exist
def create_path(directory):
    if not path.exists(directory):
        makedirs(directory)

# build path for the current image function
def build_path(directory, filename):
    return path.join(directory, filename)
