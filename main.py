from src.create_datasets import create
from models.xception_94 import xception_94
from models.vgg16 import vgg_16
from models.resnet_50 import resnet_50
from src.visualize_data import plot_training_history

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
