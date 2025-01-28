"""
sample many (10K to 100K) SIFT descriptors from the images of the training set
(you either use a detector or sample on a grid in the scale-space);
"""

import cv2
import numpy as np
import os
from sklearn.utils import shuffle

def load_images(image_fol):
    imgs = []
    for filename in os.listdir(image_fol):
        img_path = os.path.join(image_fol, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            imgs.append(img)
        else:
            print(f"Images not valid or not loaded: {filename}")

    return imgs

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, labels




def extract_sift_descriptors(images, num=10000):
    sift = cv2.SIFT_create()
    all_descriptors = []

    for img in images:
        k_points, desc = sift.detectAndCompute(img, None)

        if desc is not None:
            all_descriptors.append(desc)

    # Join all the descriptor in a unique array
    all_descriptors = np.vstack(all_descriptors)

    # sample (we need to have exactly 'num' as number of descriptors
    if len(all_descriptors) > num:
        all_descriptors = shuffle(all_descriptors)[:num]

    return all_descriptors
