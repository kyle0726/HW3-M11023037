#!/usr/bin/env python
# coding: utf-8

# # K-means

# In[96]:


import numpy as np
import matplotlib.pyplot as plt


# In[97]:


seed_num = 1
dot_num = 3916


# In[98]:


x = np.random.randint(0, 5000, dot_num)
y = np.random.randint(0, 5000, dot_num)
kx = np.random.randint(0, 5000, seed_num)
ky = np.random.randint(0, 5000, seed_num)


# In[99]:


def dis(x, y, kx, ky):
    return int(((kx-x)**2 + (ky-y)**2)**0.5)


# In[100]:


def cluster(x, y, kx, ky):
    team = []
    for i in range(1):
        team.append([])
    mid_dis = 99999999
    for i in range(dot_num):
        for j in range(seed_num):
            distant = dis(x[i], y[i], kx[j], ky[j])
            if distant < mid_dis:
                mid_dis = distant
                flag = j
        team[flag].append([x[i], y[i]])
        mid_dis = 99999999
    return team


# In[101]:


def re_seed(team, kx, ky):
    sumx = 0
    sumy = 0
    new_seed = []
    for index, nodes in enumerate(team):
        if nodes == []:
            new_seed.append([kx[index], ky[index]])
        for node in nodes:
            sumx += node[0]
            sumy += node[1]
        new_seed.append([int(sumx/len(nodes)), int(sumy/len(nodes))])
        sumx = 0
        sumy = 0
    nkx = []
    nky = []
    for i in new_seed:
        nkx.append(i[0])
        nky.append(i[1])
    return nkx, nky


# In[102]:


def kmeans(x, y, kx, ky, fig):
    team = cluster(x, y, kx, ky)
    nkx, nky = re_seed(team, kx, ky)
    
    # plot: nodes connect to seeds
    cx = []
    cy = []
    line = plt.gca()
    for index, nodes in enumerate(team):
        for node in nodes:
            cx.append([node[0], nkx[index]])
            cy.append([node[1], nky[index]])
        for i in range(len(cx)):
            line.plot(cx[i], cy[i], color='r', alpha=0.6)
        cx = []
        cy = []   
# 繪圖
kmeans(x, y, kx, ky, fig=0)
feature = plt.scatter(x, y)
k_feature = plt.scatter(kx, ky)
nk_feaure = plt.scatter(np.array(kx), np.array(ky), s=50)
plt.show()


# # 階層式分群

# In[111]:


import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering 
from sklearn import datasets

#創建數據集
X,y = datasets.make_blobs(n_samples=3916, n_features=2, random_state= 20, cluster_std = 1.5)

#Agglomerative Clustering 方法
model = AgglomerativeClustering(n_clusters = 1) 
model.fit(X) 
labels = model.fit_predict(X)
#results 可視化
plt.figure() 
plt.scatter(X[:,0], X[:,1], c = labels) 
plt.axis('equal') 
plt.title('Prediction')
plt.show()


# # DBSCAN

# In[115]:


import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs, make_circles


# In[118]:


X1, y1 = make_circles(n_samples=3916, factor=0.6, noise=0.05)
X2, y2 = make_blobs(n_samples=3916, n_features=2, centers=[[1.2, 1.2]],
                    cluster_std=[[0.1]], random_state=9)
X = np.concatenate((X1, X2))
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()


# # 階層式分群的階層樹

# In[155]:


import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


oclar = load_iris()
X = oclar.data

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# In[ ]:




