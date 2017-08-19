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
    dendogram = sch.dendrogram(sch.linkage(array, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()


if __name__ == "__main__":

    print("K Means clustering")
    data = read_file("Mall_Customers")
    X = getParams(data)
    #extracting the right amount of data
    X = X[:, [3,4]]
    DendogramMethod(X)
