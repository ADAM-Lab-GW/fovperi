import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

# Paths
DATA_ROOT = "/mnt/c/Users/zahin/Downloads/tiny-imagenet-200/tiny-imagenet-200"  # replace with the path to your dataset
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test/images")
WNIDS_PATH = os.path.join(DATA_ROOT, "wnids.txt")

# Output HDF5
OUTPUT_PATH = "imagenet.hdf5"
IMG_SIZE = 256

# Load class ids
with open(WNIDS_PATH, "r") as f:
    class_ids = [line.strip() for line in f.readlines()]
    
class_to_index = {cid: i for i, cid in enumerate(class_ids)}

# One-hot encoder
encoder = LabelBinarizer()
encoder.fit(class_ids)

# Helper to load and resize images
def load_images_labels(image_root, label_map=None):
    image_list = []
    label_list = []

    if label_map:  # TRAIN
        for class_folder in tqdm(os.listdir(image_root), desc="Processing train"):
            if class_folder not in class_ids:
                continue
            class_path = os.path.join(image_root, class_folder, "images")
            label_idx = class_to_index[class_folder]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    image_list.append(img)
                    label_list.append(class_folder)
    else:  # TEST
        for img_name in tqdm(os.listdir(image_root), desc="Processing test"):
            img_path = os.path.join(image_root, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                image_list.append(img)
                label_list.append("unknown")  # No labels in test
    return np.array(image_list, dtype=np.uint8), np.array(label_list)

# Load images and labels
train_images, train_labels = load_images_labels(TRAIN_DIR, label_map=class_to_index)
test_images, test_labels = load_images_labels(TEST_DIR)

# Convert labels to one-hot
train_labels_onehot = encoder.transform(train_labels)
test_labels_onehot = np.zeros((len(test_images), len(class_ids)), dtype=np.uint8)  # placeholder

# Save to HDF5
with h5py.File(OUTPUT_PATH, "w") as f:
    f.create_dataset("train/images", data=train_images, dtype='uint8')
    f.create_dataset("train/labels", data=train_labels_onehot, dtype='uint8')
    f.create_dataset("test/images", data=test_images, dtype='uint8')
    f.create_dataset("test/labels", data=test_labels_onehot, dtype='uint8')

print(f"HDF5 dataset created at {OUTPUT_PATH}")
