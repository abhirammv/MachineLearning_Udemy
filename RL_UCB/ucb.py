import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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

    for n in range(0, N):
        max_upper_bound = 0
        for i in range(0, d):
            if(num_times_selected[i] > 0):
                print("do something")
            else:
                upper_bound = 1e400
            if(upper_bound > max_upper_bound):
                max_upper_bound = upper_bound
                ad = i

        num_times_selected[ad] = num_times_selected[ad] + 1
        reward = dataset.values[n, ad]
        sum_rewards[ad] = sum_rewards[ad] + reward




