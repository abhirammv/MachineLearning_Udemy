import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    return dataset

def get_InOutparams(data):

    #Input:
    X = data.iloc[:, :-1].values

    #Output
    y = data.iloc[:, -1].values

    return X, y




if __name__ == "__main__":

    print("K Means clustering")
    data = read_file("Mall_Customers")
    X, y = get_InOutparams(data)


