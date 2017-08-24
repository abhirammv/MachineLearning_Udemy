import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    return dataset

if __name__ == "__main__":
    dataset = read_file("Ads_CTR_Optimisation")

    N = 10000 #number of rounds
    d = 10  #number of ads

    ads_selected = []
    numbers_of_rewards_1 = [0] * d
    numbers_of_rewards_0 = [0] * d

    for n in range(0, N):
        ad = 0
        max_random = 0
        for i in range(0, d):
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)

