import numpy as np
import os

from sklearn.preprocessing import LabelEncoder

from extract_sift import load_images, load_images_from_folder, extract_sift_descriptors
from neighbor_classification import build_confusion_matrix
from sift_clustering import kmeans_clustering
from visual_histogram import compute_visual_histogram, load_histograms


def create_data_folder():
    if not os.path.exists('data'):
        os.makedirs('data')

def save_centroids(centroids, filepath):
    np.save(filepath, centroids)
    print(f"Centroids saved to {filepath}")

def save_descriptors(desc, idx):
    filename = os.path.join(output_file_descriptors, f"descriptors_img_{idx}.npy")
    np.save(filename, desc)
    print(f"Descriptors of the image {idx} saved in {filename}")

n_samples = 10000
image_folder = "images/train/Bedroom"
output_file_centroids = "data/centroids.npy"
output_file_descriptors = "data/descriptors"
output_file_histograms_of_train = "data/histograms_of_train.npy"
output_file_histograms_of_test = "data/histograms_of_test.npy"

def main():
    create_data_folder()

    train_folder = 'images/train'
    test_folder = 'images/test'
    # all images and relative labels
    train_images, train_labels = load_images_from_folder(train_folder)
    test_images, test_labels = load_images_from_folder(test_folder)

    # SIFT DESCRIPTORS
    train_images = load_images("images/train/Bedroom")
    descriptors = extract_sift_descriptors(train_images)

    # CLUSTERING
    n_clusters = 50
    centroids, labels = kmeans_clustering(descriptors, n_clusters)
    print(f"Number of clusters: {n_clusters}")
    print(f"Shape of cluster centroids: {centroids.shape}")


    # HISTOGRAM: ho salvato solo gli istogrammi, sanza associarli a niente (labels), è giusto?
    save_centroids(centroids, output_file_centroids)
    train_histograms = compute_visual_histogram(train_images, centroids, output_file_histograms_of_train)
    histograms_train = load_histograms(output_file_histograms_of_train)
    train_histograms_array = np.array(train_histograms)

    # POINT 3
    # compute the normalized histogram for the test image to be classified (ma con i centroidi dei test o dei train?)
    train_histograms_test = compute_visual_histogram(test_images, centroids, output_file_histograms_of_test)
    histograms_test = load_histograms(output_file_histograms_of_test)
    train_histograms_test_array = np.array(train_histograms_test)

    # assign to the image the class corresponding to the training image having the closest histogram

    # ENCODING set
    encoder = LabelEncoder()
    encoded_train_labels = encoder.fit_transform(train_labels)
    encoded_test_labels = encoder.transform(test_labels)
    print(f"len of encoded_train_labels: {len(encoded_train_labels)}") # 1500
    print(f"encoded_train_labels: {encoded_train_labels}")
    print(f"len of encoded_test_labels: {len(encoded_test_labels)}") # 2985
    print(f"encoded_test_labels: {encoded_test_labels}")


    # Creare la matrice di confusione
    # cm = build_confusion_matrix(test_images, train_histograms_array, train_histograms_test_array, train_labels, test_labels)

    # print("Confusion Matrix:")
    # print(cm)


if __name__ == "__main__":
    main()
