import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

def nearest_neighbor_classification(test_histogram, train_histograms):
    # Calcolare la distanza euclidea tra l'istogramma di test e gli istogrammi di addestramento
    distances = cdist([test_histogram], train_histograms, metric='euclidean')
    return np.argmin(distances)  # Restituisci l'indice dell'immagine di addestramento più vicina




def build_confusion_matrix(test_images, train_histograms, test_histograms, labels_train, labels_test):
    print("labels_test: ", labels_test)
    print("test_histograms: ", test_histograms)
    predictions = []
    true_labels = []

    for i, test_hist in enumerate(test_histograms):
        predicted_label = nearest_neighbor_classification(test_hist, train_histograms)

        # add the prevision le predictions and the real label
        predictions.append(predicted_label)
        true_labels.append(labels_test[i])  # real label for the test images

    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    return cm
