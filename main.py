

#librerie
import os
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import random as rd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
#from yellowbrick.cluster import SilhouetteVisualizer
import carico_immagini as ci
import visual_vocabolary as vv
import classificazioni as clf

#carico immagini training
cartella_train = '/home/eva/Scrivania/esame_cv/train' 
images_training, train_labels  = ci.carica_immagini_e_etichette(cartella_train)
labels = np.unique(np.array(train_labels)) # nomi classi

#carico immagini test
cartella_test = '/home/eva/Scrivania/esame_cv/test' 
images_test, test_labels  = ci.carica_immagini_e_etichette(cartella_test)

'''
punto 1 

build a visual vocabulary:
*   sample many (10K to 100K) SIFT descriptors from the images of
the training set (you either use a detector or sample on a grid in the
scale-space);
*  cluster them using k-means (the choice of the number of clusters is
up to you, and you should experiment with different values, but you
could start with a few dozens);
*  collect (and save for future use) the clusters’ centroids which repre-
sent the k 128-dimensional visual words.

'''

#calcolo i descrittori dei key-points
all_descriptors = vv.find_descriptors(100, images_training) 
all_descriptors = np.vstack(all_descriptors)
print(np.shape(all_descriptors))

#ottengo le visual words ---  da fare tuning 
k = 24
visual_words = vv.centroids(all_descriptors,k)
print(np.shape(visual_words))


''' 
punto 2 
Represent each image of the training set as a normalized histogram having
k bins, each corresponding to a visual word; a possibility is to perform a
rather dense sampling in space and scale; another possibility is to use the
SIFT detector to find the points in scale-space where the descriptor is
computed. In any case, each computed descriptor will increase the value
of the bin corresponding to the closest visual word.
'''
#istrogrammi training
training_hist = compute_histograms(images_training)

'''
punto 3
Employ a nearest neighbor classifier and evaluate its performance:
• compute the normalized histogram for the test image to be classified;
• assign to the image the class corresponding to the training image
having the closest histogram.
• repeat for all the test images and build a confusion matrix.
'''

#istrogrammi test
test_hist = compute_histograms(images_test)

#classificazione istogrammi
predict_labels = clf.test_assign_vv(training_hist, test_hist)
#converto in text 
predict_labels = [labels[prediction] for prediction in predict_labels]

#confusion matrix
cm = metrics.confusion_matrix(test_labels, predicted_labels)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_labels + predicted_labels))
cm_display.plot()
im = cm_display.im_
im.set_cmap('YlGnBu')
plt.show()


'''
punto 4
Train a multiclass linear Support Vector Machine, using the one-vs-rest
approach (you will need to train 15 binary classifiers having the normalized
histograms as the input vectors and positive labels for the “one” class and
negative for the “rest.”)
'''
training_hist = np.array(training_hist)
classifier_svm = clf.one_vs_all_svm_train(training_hist, train_labels)

'''
punto 5 
Evaluate the multiclass SVM:
• compute the normalized histogram for the test image to be classified;
• compute the real-valued output of each of the SVMs, using that his-
togram as input;
• assign to the image the class corresponding to the SVM having the
greatest real-valued output.
• repeat for all the test images and build a confusion matrix.
'''
#calcolati al punto 3
test_hist = np.array(test_hist) 

#classificazione test images
test_svm_pred = clf.one_vs_all_svm_test(classifier_svm, test_hist)

#converto in text per semplicità di lettura
test_svm_pred = [labels[prediction] for prediction in test_svm_pred]

#cnfusion matrix
cm = metrics.confusion_matrix(test_labels, predicted_labels)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_labels + predicted_labels))
cm_display.plot()
im = cm_display.im_
im.set_cmap('YlGnBu')
plt.show()

