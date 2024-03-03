from src.image_analysis.data_preprocessing import ImagePreprocessor
from os import path

target_image_size = (128, 128)
categories=['NORMAL', 'PNEUMONIA']
data_types=['test', 'train', 'val']

preprocessor = ImagePreprocessor(target_size=target_image_size)
preprocessor.process_dataset()
del preprocessor