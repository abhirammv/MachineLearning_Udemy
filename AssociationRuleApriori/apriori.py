import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name), header=None);
    return dataset


if __name__ == "__main__":
    dataset = read_file("Market_Basket_Optimisation")
    #No need to extract parameters using .iloc
    #The below is the way of putting each transaction into a list

    transactions = []
    for i in range(0, 7501):
        transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

    from apyori import apriori
    rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

    results = list(rules)





