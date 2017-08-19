import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    return dataset

def getParams(data):
    X = data.iloc[:,:].values
    return X

def ElbowMethod(array):
    #This method helps in determining the optimum K value in K Means Clustering algorithm
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10)
        kmeans.fit(array)
        wcss.append(kmeans.inertia_)
    #Plotting values of WCSS - Within Cluster Sum of Sqaures
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def ClusterData(array, k):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, init="k-means++")
    cl_array = kmeans.fit_predict(array)
    return cl_array, kmeans

def Visualize(array, clusters, k_meansAlg):
    plt.scatter(array[k_meansAlg == 0, 0], array[k_meansAlg == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(array[k_meansAlg == 1, 0], array[k_meansAlg == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(array[k_meansAlg == 2, 0], array[k_meansAlg == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(array[k_meansAlg == 3, 0], array[k_meansAlg == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(array[k_meansAlg == 4, 0], array[k_meansAlg == 4, 1], s=100, c='magenta', label='Cluster 5')
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
    ElbowMethod(X)
    k = int(input("Input the K value: "))
    data_clusters, k_meansAlg = ClusterData(X, k)









