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

def PolynomialReg(in_data, out_data, deg=2):
    from sklearn.preprocessing import PolynomialFeatures
    data_modifier = PolynomialFeatures(degree=2)
    in_data_poly = data_modifier.fit_transform(in_data)

    #Considering in_data_poly as the new independent variable matrix, perform linear regression
    poly_reg = LinReg(in_data_poly, out_data)
    return poly_reg

if __name__ == "__main__":
    dataset = read_file("Position_Salaries")
    X, y = get_InOutparams(dataset)

    #Ignoring the Position Parameter in this dataset
    X = X[:, 1:]

    #Fitting the linear regression model
    regressor_lin = LinReg(X, y)

    #Fitting the polynomial regression
    regressor_poly = PolynomialReg(X, y, deg=4)








