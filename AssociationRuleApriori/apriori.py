import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    return dataset

def getParams(data):
    X = data.iloc[:, :].values
    return X



if __name__ == "__main__":
    data = read_file("Market_Basket_Optimisation")

