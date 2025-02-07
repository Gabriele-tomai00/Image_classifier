import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#SIFT descriptor
def find_descriptors(k_p, images):
    '''
    SIFT descriptor per trovare i key point 
    '''
    all_descriptors = []
    for image_path in images:
      image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
      sift = cv2.SIFT_create(k_p)
      keypoints, descriptors = sift.detectAndCompute(image, None)
      if descriptors is not None:
        all_descriptors.append(descriptors)

    return  np.vstack(all_descriptors)

#Cluster descriptors per visual words (vocabulary)
#tuning necessario
def centroids(descriptors, k):
  '''
  trova i centroidi dei clusters corrispondenti alle visual words
  '''
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(descriptors) #---> in input qui ogni descriptor di 128 è un punto da classificare
  visual_words = kmeans.cluster_centers_
  return visual_words

#per trovare il corretto numero di visual words uso due  metriche per verificare il numero ottimale di k 
def find_numbers_centroids(k_list, descriptors):
  '''
  per trovare il numero ideale di visual words uso il silhoutte score
  '''
  inertia_values = []
  silhouette_scores = []
  k_list = [x for x in range(12,120,12)]
  for k in k_list:
    print(("k= {}").format(k))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(descriptors)
    inertia_values.append(kmeans.inertia_)
    score = silhouette_score(descriptors, kmeans.labels_)
    silhouette_scores.append(score)

  # Plot inertia values
  plt.plot(k_values, inertia_values, marker='o')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Inertia')
  plt.title('Elbow Method for Optimal k')
  plt.show()
   
  # Plot silhouette scores
  plt.plot(k_values, silhouette_scores, marker='o')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Silhouette Score')
  plt.title('Silhouette Score for Different k')
  plt.show()


def compute_histograms(images):
  '''
  per ottenere gli istogrammi per ogni immagine
  '''
  histograms = []
  for image_path in images:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    hist = np.zeros(k)
    if descriptors is not None:
        for descriptor in descriptors:
            distances = np.linalg.norm(visual_words - descriptor, axis=1) #forse va cambiata
            visual_word_index = np.argmin(distances)
            hist[visual_word_index] += 1

    hist = hist / np.sum(hist)  #normalizzo l'istogramma
    histograms.append(hist)
  return histograms