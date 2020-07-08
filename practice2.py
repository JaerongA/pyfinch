# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# from sklearn import decomposition
# from sklearn import datasets
#
# np.random.seed(5)
#
# centers = [[1, 1], [-1, -1], [1, -1]]
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
# plt.cla()
# pca = decomposition.PCA(n_components=3)
# pca.fit(X)
# X = pca.transform(X)


import pandas as pd
import numpy as np
import random as rd

import matplotlib.pyplot as plt

a = [1,2,3]
b = [2,3,4]
fig = plt.figure()
plt.plot(a,b)
plt.show()
plt.close(fig)
