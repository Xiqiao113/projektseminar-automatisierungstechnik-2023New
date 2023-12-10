import torch
import numpy as np
from sklearn.cluster import KMeans

mode_views = torch.load('mode_view.t')

chromosome = np.load('test.npy')

mode_views = torch.reshape(mode_views, (mode_views.shape[0], -1))

kmeans = KMeans(n_clusters=4, n_init='auto')
kmeans.fit(mode_views)

new_mode = kmeans.cluster_centers_.reshape(-1)
chromosome[:4*11**2] = new_mode
np.save('new_test.npy', chromosome)