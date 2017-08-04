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

def FeatureScaler(array):
    from sklearn.preprocessing import StandardScaler
    SS = StandardScaler()
    array = SS.fit_transform(array)
    return array

def TrainTestSplit(array):
    from sklearn.cross_validation import train_test_split
    array_tr, array_te = train_test_split(array, test_size=0.2, random_state=0)
    return array_tr, array_te

def LogReg(in_data, out_data):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(in_data, out_data)
    return classifier

def Pred(test_data, model):
    pred = model.predict(test_data)
    return pred

if __name__ == "__main__":
    dataset = read_file("Social_Network_Ads")
    X, y = get_InOutparams(dataset)
    X = X[:, 2:4] #Selection of parameters
    X = FeatureScaler(X)
    X_tr, X_te = TrainTestSplit(X)
    y_tr, y_te = TrainTestSplit(y)
    LogRegClassifier = LogReg(X_tr, y_tr)


