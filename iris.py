#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


# In[130]:


iris = pd.read_csv("iris.csv")
x = iris.iloc[:, [0, 1, 2, 3,]].values


# In[131]:


iris.info()
iris[0:10]


# In[132]:


iris_outcome = pd.crosstab(index=iris["class"],  
                              columns="count")     

iris_outcome


# In[133]:


iris_setosa=iris.loc[iris["class"]=="Iris-setosa"]
iris_virginica=iris.loc[iris["class"]=="Iris-virginica"]
iris_versicolor=iris.loc[iris["class"]=="Iris-versicolor"]


# In[134]:


sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"sepal_width").add_legend()
plt.show()


# In[135]:


sns.set_style("whitegrid")
sns.pairplot(iris,hue="class",size=3);
plt.show()


# In[136]:


sns.boxplot(x="class",y="petal_length",data=iris)
plt.show()


# In[137]:


sns.violinplot(x="class",y="petal_length",data=iris)
plt.show()


# In[138]:


from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[139]:


plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


# In[140]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[141]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')

plt.legend()


# In[142]:


fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.show()


# In[143]:


dist_sin = linkage(iris.iloc[:, [0, 1, 2, 3]],method="complete")
plt.figure(figsize=(18,6))
dendrogram(dist_sin, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("DENDROGRAM COMPLETE METHOD",fontsize=18)
plt.show()


# In[144]:


import scipy.stats                
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage


# In[145]:


import os
print(os.listdir("../ppython"))


# In[146]:


print(iris.head())


# In[147]:


print(iris.shape)


# In[148]:


matcorr = iris.iloc[:,~iris.columns.isin(['','class'])].corr()
mask = np.zeros_like(matcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(matcorr, mask=mask, cmap="Blues", vmin=-1, vmax=1, center=0, square=True);
plt.show()


# In[149]:


g = sns.PairGrid(iris.iloc[:, [0, 1, 2, 3]])
g.map_diag(plt.hist, histtype="step", linewidth=4)
g.map_offdiag(plt.scatter)


# In[150]:


dist_sin = linkage(iris.iloc[:, [0, 1, 2, 3]],method="single")
plt.figure(figsize=(18,6))
dendrogram(dist_sin, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("DENDROGRAM SINGLE METHOD",fontsize=18)
plt.show()


# In[151]:


from scipy.cluster.hierarchy import fcluster
iris_SM=iris.copy()

iris_SM['2_clust']=fcluster(dist_sin,2, criterion='maxclust')
iris_SM['3_clust']=fcluster(dist_sin,3, criterion='maxclust')
iris_SM.head()


# In[152]:


plt.figure(figsize=(24,4))

plt.suptitle("Hierarchical Clustering Single Method",fontsize=18)

plt.subplot(1,3,1)
plt.title("K = 2",fontsize=14)
sns.scatterplot(x="petal_length",y="petal_width", data=iris_SM, hue="2_clust")

plt.subplot(1,3,2)
plt.title("K = 3",fontsize=14)
sns.scatterplot(x="petal_length",y="petal_width", data=iris_SM, hue="3_clust")

plt.subplot(1,3,3)
plt.title("class",fontsize=14)
sns.scatterplot(x="petal_length",y="petal_width", data=iris_SM, hue="class")


# In[153]:


plt.figure(figsize=(24,4))
plt.subplot(1,2,1)
plt.title("K = 2",fontsize=14)
sns.swarmplot(x="class",y="2_clust", data=iris_SM, hue="class")

plt.subplot(1,2,2)
plt.title("K = 3",fontsize=14)
sns.swarmplot(x="class",y="3_clust", data=iris_SM, hue="class")


# In[154]:


y = iris_SM.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values


# In[155]:


sns.heatmap(iris_SM.iloc[:, [0, 1, 2, 3, 4, 5]].groupby(['2_clust']).mean(), cmap="Purples")


# In[156]:


g = sns.PairGrid(iris_SM, vars=["sepal_width","petal_length","petal_width"], hue='2_clust')
g.map(plt.scatter)
g.add_legend()


# In[157]:


dist_comp = linkage(iris_SM.iloc[:, [0, 1, 2, 3]],method="complete")

plt.figure(figsize=(18,6))
dendrogram(dist_comp, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("DENDROGRAM COMPLETE METHOD",fontsize=18) 
plt.show()


# In[158]:


iris_CM=iris.copy()
iris_CM['2_clust']=fcluster(dist_comp,2, criterion='maxclust')
iris_CM['3_clust']=fcluster(dist_comp,3, criterion='maxclust')
iris_CM.head()


# In[159]:


plt.figure(figsize=(24,4))

plt.suptitle("Hierarchical Clustering Complete Method",fontsize=9)

plt.subplot(1,3,1)
plt.title("K = 2",fontsize=14)
sns.scatterplot(x="petal_length",y="sepal_width", data=iris_CM, hue="2_clust")

plt.subplot(1,3,2)
plt.title("K = 3",fontsize=14)
sns.scatterplot(x="petal_length",y="sepal_width", data=iris_CM, hue="3_clust")

plt.subplot(1,3,3)
plt.title("K = 4",fontsize=14)
sns.scatterplot(x="petal_length",y="sepal_width", data=iris_CM, hue="class")


# In[160]:


plt.figure(figsize=(24,4))
plt.subplot(1,2,1)
plt.title("K = 2",fontsize=14)
sns.swarmplot(x="class",y="2_clust", data=iris_CM, hue="class")

plt.subplot(1,2,2)
plt.title("K = 3",fontsize=14)
sns.swarmplot(x="class",y="3_clust", data=iris_CM, hue="class")


# In[161]:


print(pd.crosstab(iris_CM["class"],iris_CM["3_clust"]))


# In[162]:


sns.heatmap(iris_CM.iloc[:,[0, 1, 2, 3, 4, 6]].groupby(['3_clust']).mean(), cmap="Purples")


# In[163]:


g = sns.PairGrid(iris_CM, vars=["sepal_width","petal_length","petal_width"], hue='3_clust')
g.map(plt.scatter)
g.add_legend()


# In[164]:


dist_comp = linkage(iris.iloc[:, [0, 1, 2, 3]],method="ward")

plt.figure(figsize=(18,6))
dendrogram(dist_comp, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("DENDROGRAM COMPLETE METHOD",fontsize=18) 
plt.show()


# In[165]:


iris_WM=iris.copy()
iris_WM['2_clust']=fcluster(dist_comp,2, criterion='maxclust')
iris_WM['3_clust']=fcluster(dist_comp,3, criterion='maxclust')
iris_WM['4_clust']=fcluster(dist_comp,4, criterion='maxclust')
iris_WM.head()


# In[166]:


plt.figure(figsize=(24,4))

plt.suptitle("Hierarchical Clustering Complete Method",fontsize=9)

plt.subplot(1,4,1)
plt.title("K = 2",fontsize=14)
sns.scatterplot(x="sepal_width",y="petal_width", data=iris_WM, hue="2_clust",palette="Paired")

plt.subplot(1,4,2)
plt.title("K = 3",fontsize=14)
sns.scatterplot(x="sepal_width",y="petal_width", data=iris_WM, hue="3_clust",palette="Paired")

plt.subplot(1,4,3)
plt.title("K = 4",fontsize=14)
sns.scatterplot(x="sepal_width",y="petal_width", data=iris_WM, hue="4_clust",palette="Paired")

plt.subplot(1,4,4)
plt.title("class",fontsize=14)
sns.scatterplot(x="sepal_width",y="petal_width", data=iris_WM, hue="class")


# In[167]:


plt.figure(figsize=(24,4))
plt.subplot(1,3,1)
plt.title("K = 2",fontsize=14)
sns.swarmplot(x="class",y="2_clust", data=iris_WM, hue="class")

plt.subplot(1,3,2)
plt.title("K = 3",fontsize=14)
sns.swarmplot(x="class",y="3_clust", data=iris_WM, hue="class")

plt.subplot(1,3,3)
plt.title("K = 4",fontsize=14)
sns.swarmplot(x="class",y="4_clust", data=iris_WM, hue="class")


# In[168]:


print(pd.crosstab(iris_CM["class"],iris_WM["3_clust"]))
print('_____________________________________________')
print(pd.crosstab(iris_CM["class"],iris_WM["4_clust"]))


# In[169]:


plt.figure(figsize=(24,4))

plt.subplot(1,3,1)
plt.title("K = 2",fontsize=14)
sns.heatmap(iris_WM.iloc[:,[0, 1, 2, 3, 4, 5]].groupby(['2_clust']).mean(), cmap="Purples")

plt.subplot(1,3,2)
plt.title("K = 3",fontsize=14)
sns.heatmap(iris_WM.iloc[:,[0, 1, 2, 3, 4, 6]].groupby(['3_clust']).mean(), cmap="Purples")

plt.subplot(1,3,3)
plt.title("K = 4",fontsize=14)
sns.heatmap(iris_WM.iloc[:,[0, 1, 2, 3, 4, 7]].groupby(['4_clust']).mean(), cmap="Purples")


# In[170]:


g = sns.PairGrid(iris_WM, vars=["sepal_width","petal_length","petal_width"], hue='4_clust')
g.map(plt.scatter)
g.add_legend()


# In[171]:


from sklearn.cluster import DBSCAN


# In[172]:


TRAIN_LABEL_COL = "class"
TRAIN_FEATURES = [col for col in iris.columns if col != TRAIN_LABEL_COL]
X = iris[TRAIN_FEATURES]
y = iris[TRAIN_LABEL_COL]


# In[173]:


dbscan = DBSCAN(eps=0.5,min_samples=5)
y_pred = dbscan.fit_predict(X)


# In[174]:


dbscan.labels_


# In[175]:


np.unique(dbscan.labels_)


# In[176]:


len(dbscan.core_sample_indices_)


# In[177]:


dbscan.core_sample_indices_[:10]


# In[178]:


dbscan.components_[:3]


# In[179]:


y_pred_df = pd.DataFrame(y_pred)
y_pred_df.columns=['pred_label']

val = pd.concat([X,y_pred_df],axis=1)

print(val)


# In[180]:


sns.color_palette("pastel")
sns.pairplot(val,hue='pred_label')
plt.show()


# In[181]:


from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


# In[182]:


dist_comp = linkage(iris.iloc[:, [0, 1, 2, 3]],method="ward")

plt.figure(figsize=(18,6))
dendrogram(dist_comp, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("DENDROGRAM WARD METHOD",fontsize=18) 
plt.show()


# In[ ]:




