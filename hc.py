# Data Preprocessing Template
#Hierarchical Clustering
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values



#Visualizing dendogram to help with clustering
import scipy.cluster.hierarchy as sch #library for hierarchal clustering
dendrogram = sch.dendrogram(sch.linkage(X, 'ward')) #getting dendogram and ward parameter minimizes variance between clusters
plt.title('Dendogram')
plt.ylabel('Euclidian Distance')
plt.xlabel('Customers')
plt.show()

#Fitting hierarchal clustering to dataset once analyzed dendrogram 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
#after looking at dendrogram, we can see 5 is optimal clusters and distance we would like to measure by is euclidean

y_hc = hc.fit_predict(X)

#visualize cluster

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, color = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, color = 'black', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, color = 'blue', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, color = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, color = 'green', label = 'Cluster 5')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, color = 'brown', label = 'Cluster Centers')
plt.legend()
plt.show()