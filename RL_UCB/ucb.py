import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name), header=None);
    return dataset

if __name__=="__main__":
    print("Upper confidence bound")