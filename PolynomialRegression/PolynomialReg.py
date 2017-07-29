#Implementation of Polynomial Regression
#The procedure is is similar to Linear Regression
#Polynomial regression is used when the dataset is not linear and must be fit better by a non-linear function

import numpy
import pandas as pd
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

def LinReg(in_data, out_data):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(in_data, out_data)
    return regressor

def PolynomialReg(in_data, out_data):
    from sklearn.preprocessing import PolynomialFeatures
    regressor = PolynomialFeatures(degree=2) #For modifying the independent variable
    in_data_poly = regressor.fit_transform(in_data)
    regressor.fit(in_data_poly, out_data)






if __name__ == "__main__":
    dataset = read_file("Position_Salaries")
    X, y = get_InOutparams(dataset)

    #Ignoring the Position Parameter in this dataset
    X = X[:, 1:]

    #Fitting the linear regression model
    regressor_lin = LinReg(X, y)
    from sklearn.preprocessing import PolynomialFeatures
    reg = PolynomialFeatures(degree=4)
    X_poly = reg.fit_transform(X)





