import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    return dataset

def getParams(data):
    X = data.iloc[:, :].values
    return X

def DendogramMethod(array):
    import scipy.cluster.hierarchy as sch
    #ward method helps in determining the within cluster variance
    dendogram = sch.dendrogram(sch.linkage(array, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

def ClusterData(array, n):
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    cl_array = hc.fit_predict(array)
    return cl_array, hc

def visualize(array, clusters, hc_alg):
    plt.scatter(array[clusters == 0, 0], array[clusters == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(array[clusters == 1, 0], array[clusters == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(array[clusters == 2, 0], array[clusters == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(array[clusters == 3, 0], array[clusters == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(array[clusters == 4, 0], array[clusters == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(k_meansAlg.cluster_centers_[:, 0], k_meansAlg.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    print("K Means clustering")
    data = read_file("Mall_Customers")
    X = getParams(data)
    #extracting the right amount of data
    X = X[:, [3,4]]
    DendogramMethod(X)
    n = int(input("Input the number of clusters: "))
    data_clusters, hc_alg = ClusterData(X, n)
    visualize(array=X, clusters=data_clusters, hc_alg=hc_alg)

