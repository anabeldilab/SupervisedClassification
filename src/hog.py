from skimage.feature import hog
import cv2


def HOG(data, size=(128, 128), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1)):
    image_path_array = data['filepath']
    labels = data['label']
    feature_data = []
    for image_path in image_path_array:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Cant read image: {image_path}')
            return None, None

        image = cv2.resize(image, size)
        fd = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block, visualize=False)
        print("HOG feature extraction image_path: ", image_path)
        feature_data.append(fd)

    HOG_data = {
        'features': feature_data,
        'label': labels
    }

    return HOG_data


