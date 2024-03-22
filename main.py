from src.create_datasets import create, load_data
from models.xception_94 import xception_94
from models.vgg16 import vgg_16
from models.resnet_50 import resnet_50
from models.svm import svm_classifier as svm
from src.visualize_data import plot_training_history
from src.hog import HOG

# Load the dataset
path = 'data'
train_dir = 'data/train'
test_dir = 'data/test'
train_ds, validation_ds, test_ds = create(path, train_dir, test_dir)

# Train the model xception
xception_history = xception_94(train_ds, validation_ds, test_ds)

# Train the model vgg16
vgg_16_history = vgg_16(train_ds, validation_ds, test_ds)

# Train the model resnet_50
resnet_50_history = resnet_50(train_ds, validation_ds, test_ds)

# Plot the training history
plot_training_history(xception_history)
plot_training_history(vgg_16_history)
plot_training_history(resnet_50_history)

##########  HOG + SVM  ##########
# Load the dataset
train_data = load_data(train_dir)
test_data = load_data(test_dir)

# Extract the features
train_features = HOG(train_data['filepath'])
train_data['features'] = train_features

test_features = HOG(test_data['filepath'])
test_data['features'] = test_features

# Build the model
cv_metrics, test_metrics = svm(train_data, test_data)



