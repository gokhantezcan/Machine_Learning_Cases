#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:03:29 2019

@author: gokhan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3), axis = 0)
y = np.concatenate((y1,y2,y3), axis = 0)

dictionary = {"x":x, "y":y}

data = pd.DataFrame(dictionary)

"""
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x3, y3)
plt.show()
"""

#dendogram
from scipy.cluster.hierarchy import linkage, dendrogram
# linkage dendogramdır aslında dendogramı çizdirmek için kullanacağım hier. clustering algoritmam
 
merg = linkage(data, method="ward") # bizim kmeans te kullandığımız wcss buda aynı sekilde
# cluster içindeki varyansları kucultecek sekilde algortimayı sekillendiren birsey
#yayılımları minimalize eder.

dendrogram(merg,leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("Euclidian Distance")
plt.show()


from sklearn.cluster import AgglomerativeClustering  # herbir datapointtten birbiriyle en yakınları kullanarak
# tek bir cluster a ulaşmayı sağlar.
hierartivcal_cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean",linkage="ward")
cluster = hierartivcal_cluster.fit_predict(data) # fit modeli oluştur ve daha sonra datama göre prediction yap
# ve clusterlarımı oluştur.

data["label"] = cluster
"""
plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = "red")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = "green")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = "blue")
plt.show()
"""


















