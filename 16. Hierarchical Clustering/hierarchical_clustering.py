# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing mall dataset with pandas
dataset = pd.read_csv('Mall.csv')
X = dataset.iloc[:, [3, 4]].values

# Using a dendrogram to find the optimal number of cluster
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucliden distances')
plt.show()