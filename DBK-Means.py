import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import time
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import plotly.graph_objs as go


import pandas_helper_calc

df = pd.read_csv('loans.csv') #data to be read

#get only the dimensions you need. IMPORTANT: do not use data that classifies someone or strings, such as eye color, dates, education etc
X = df.drop(['Loan_ID', 'effective_date', 'due_date', 'paid_off_time', 'past_due_days', 'education', 'loan_status', 'Gender'], axis=1)
X = X[:299]

#min samples is minimum amount of samples for a cluster to be created, currently set to double the feature variables
min_samples = X.shape[1]*2

#db scan implementation, only change eps here, change it with the code at the very bottom
db = DBSCAN(eps=3, min_samples=min_samples).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

index = [i for i, x in enumerate(list(labels)) if x == -1]

#print anomalous data points
print(df.iloc[[*index]])

#if the dimensions is greater than 2 decrese to 2, this is only for viewing purposes
if X.shape[1] > 2:
    X_tsne = pd.DataFrame(TSNE(n_components=2, random_state=0).fit_transform(X))

#show the plot with all datapoints and coloured clusters
plt.scatter(X_tsne.iloc[:,0], X_tsne.iloc[:,1], c=labels, cmap='rainbow')
plt.colorbar()
plt.title("DBSCAN")
plt.show()


from sklearn.cluster import KMeans

#implement k-means
kmeans = KMeans(n_clusters=n_clusters_, random_state=0).fit(X)

#predict for data point(s), to add more add more rows to the pd dataframe ex: {0:[1000, 10],1:[15, 10],2:[74, 10]}, currently using test data
prediction = kmeans.predict(pd.DataFrame({0:[1000],1:[15],2:[74]}))
#k-means anomaly is group 0 while db scan is -1, so just subtract 1 to get the db scan group that you can see in the plot
print(prediction-1)

#show plot again. simply for viewing purposes
plt.scatter(X_tsne.iloc[:,0], X_tsne.iloc[:,1], c=labels, cmap='rainbow')
plt.colorbar()
plt.title("DBSCAN")
plt.show()


#uncomment this code to find the optimal eps value for db scan, when you do you will get a graph, and find the point with maximum curvature, set eps to the y value of that point, or position on line
"""neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distLen = len(distances)

distances = np.sort(distances, axis=0)
distances = distances[:,1]

plt.plot(distances)

plt.show()"""
