import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Default values
radius = 3
points = 8 * radius

def LBP(data, size=(128, 128), radius=radius, points=points):
    image_path_array = data['filepath']
    labels = data['label']
    feature_data = []
    for image_path in image_path_array:
        imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            print(f'Cant read image: {image_path}')
            return None

        imagen = cv2.resize(imagen, size)
        lbp = local_binary_pattern(imagen, points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                bins=np.arange(0, points + 3),
                                range=(0, points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        print("LBP feature extraction image_path: ", image_path)
        feature_data.append(hist)

    LBP_data = {
        'features': feature_data,
        'label': labels
    }
    return LBP_data