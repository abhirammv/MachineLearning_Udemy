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
    #Plotting




if __name__ == "__main__":

    print("K Means clustering")
    data = read_file("Mall_Customers")
    X = getParams(data)
    #extracting the right amount of data
    X = X[:, [3,4]]




