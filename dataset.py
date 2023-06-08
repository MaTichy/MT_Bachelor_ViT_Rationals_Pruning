import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from helpers import download_data, create_dataloaders

import os
from os import getcwd

import random
import numpy as np

# Set the batch size
# 32 # this is lower than the ViT paper but it's because we're starting small
BATCH_SIZE = 256 #64
seed = 42   #42

# Create image size (from Table 3 in the ViT paper) 
IMG_SIZE = 64 # 64

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# Download tiny imagenet
image_path = download_data(source="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
                           destination="tiny_imagenet_mt")

current = getcwd()
#image_path = os.path.join(current, image_path )

image_path = current / image_path

image_path = image_path / "tiny-imagenet-200"

# Setup directory paths to train and test images
train_data = image_path / "train"
test_data = image_path / "test"
valid_data_dir = image_path / "val"

val_img_dir = os.path.join(valid_data_dir, 'images')

# Open and read val annotations text file
fp = open(os.path.join(valid_data_dir, 'val_annotations.txt'), 'r')
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
valid_data = {}
for line in data:
    words = line.split('\t')
    valid_data[words[0]] = words[1]
fp.close()

for img, folder in valid_data.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

val_dir = val_img_dir

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])           
print(f"Manually created transforms: {manual_transforms}")

# Create data loaders
train_loader, test_loader, valid_loader, class_names = create_dataloaders(
    train_dir=train_data,
    test_dir=test_data,
    val_dir=val_dir,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)


print(len(train_loader), len(test_loader), len(valid_loader))

"""
# Get a batch of images
image_batch, label_batch = next(iter(train_loader))


# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# View the batch shapes
#print(image.shape, label)

# Plot image with matplotlib
plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
plt.title(class_names[label])
plt.show()
"""
