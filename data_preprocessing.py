# Below is the code for loading images, resizing, normalizing, and augmenting using OpenCV.
import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize
            images.append(img)
    return np.array(images)

# Example usage:
# train_images = load_images_from_folder('data/train')

# For augmentation, you can use OpenCV functions like flip, rotate, etc.
# Example:
# img_flip = cv2.flip(img, 1)  # Horizontal flip
