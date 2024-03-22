from skimage.feature import hog
import cv2


def HOG(image_path_array, size=(128, 128)):
    feature_data = []
    for image_path in image_path_array:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Cant read image: {image_path}')
            return None, None

        image = cv2.resize(image, size)
        fd = hog(image, orientations=8, pixels_per_cell=(8, 8),
                cells_per_block=(1, 1), visualize=False)
        print("image_path: ", image_path)
        feature_data.append(fd)

    return feature_data


