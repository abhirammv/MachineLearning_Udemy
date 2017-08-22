import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name), header=None);
    return dataset

if __name__=="__main__":
    dataset = read_file("Ads_CTR_Optimisation")
    #There are no X and y parameters

    N = 10000 #number of rounds
    d = 10  #number of ads

    num_times_selected = [0] * d
    sum_rewards = [0] * d


