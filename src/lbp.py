import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Default values
radio = 3
n_puntos = 8 * radio

def LBP(image_path_array, size=(128, 128), n_puntos=n_puntos, radio=radio):
    feature_data = []
    for image_path in image_path_array:
        imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            print(f'Cant read image: {image_path}')
            return None

        imagen = cv2.resize(imagen, size)
        lbp = local_binary_pattern(imagen, n_puntos, radio, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                bins=np.arange(0, n_puntos + 3),
                                range=(0, n_puntos + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        feature_data.append(hist)

    return feature_data