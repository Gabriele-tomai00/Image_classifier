import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2

def save_histograms(histograms, output_file_histograms):
    np.save(output_file_histograms, histograms)  # Salva solo gli istogrammi come array

def load_histograms(output_file_histograms):
    histograms = np.load(output_file_histograms, allow_pickle=True)  # Carica l'array degli istogrammi
    return histograms

def compute_visual_histogram(images, centroids, output_file_histograms):
    sift = cv2.SIFT_create()
    k = centroids.shape[0]  # Number of visual words
    histograms = []

    for img in images:  # Supponendo che 'images' sia una lista di tuple (immagine, etichetta)
        step_size = 8  # Distance between the grid points
        keypoints = [cv2.KeyPoint(x, y, step_size)
                     for y in range(0, img.shape[0], step_size)
                     for x in range(0, img.shape[1], step_size)]
        _, descriptors = sift.compute(img, keypoints)

        if descriptors is None:
            histograms.append(np.zeros(k))
            continue

        # Find the closest centroid for every descriptor
        nbrs = NearestNeighbors(n_neighbors=1).fit(centroids)
        distances, indices = nbrs.kneighbors(descriptors)

        # Create the histogram
        histogram = np.zeros(k)
        for idx in indices:
            histogram[idx[0]] += 1

        # Normalize the histogram
        histogram /= np.sum(histogram)
        histograms.append(histogram)

    if output_file_histograms:
        save_histograms(histograms, output_file_histograms)


