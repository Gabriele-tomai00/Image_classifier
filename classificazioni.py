
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import random as rd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def test_assign_vv(train, test):
    '''
    per assegnare le immagini di test alla visual word più vicina
    '''
    knn = NearestNeighbors(n_neighbors=10) #numero da verificare ---> puoi usare altri algoritmi
    knn.fit(train)
    predictions = knn.kneighbors(test, return_distance=False)
    return predictions

def one_vs_all_svm_train(train, train_labels):
    '''
    fase di training di 15 SVM una per classe
    '''
    rd.shuffle(train)
    #labels binarie
    y_train_bin = []
    for obj in train_labels:
        binary_row = [1 if obj == u_obj else 0 for u_obj in np.unique(train_labels)]
        y_train_bin.append(binary_row)
    y_train_bin = np.array(y_train_bin)

    y_train = np.array(train_labels)
    n_classes = len(np.unique(y_train)) #numero delle classi 

    classifiers = []
    for i in range(n_classes):
        svm = SVC(kernel='linear', probability=True)
        svm.fit(train, y_train_bin[:,i])
        classifiers.append(svm)
    return classifiers

def one_vs_all_svm_test(classifiers, test):
    #test
    rd.shuffle(test)

    predictions = []
    # scores ha come righe le immagini di test, come colonne i classificatori, uso il trasposto per calcolare il max
    scores = np.array([clf.decision_function(X_test) for clf in classifiers]).T
    for i in range(len(X_test)):
        predicted_label = np.argmax(scores[i])
        predictions.append(predicted_label)

    return predictions
