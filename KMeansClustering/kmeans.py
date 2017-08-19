import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    return dataset

def getParams(data):
    X = data.iloc[:,:].values
    return X




if __name__ == "__main__":

    print("K Means clustering")
    data = read_file("Mall_Customers")
    X = getParams(data)
    #extracting the right amount of data
    X = X[:, [3,4]]



