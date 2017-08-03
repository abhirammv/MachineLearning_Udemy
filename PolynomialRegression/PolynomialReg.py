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

def PolynomialFeatures(in_data, deg=2):
    from sklearn.preprocessing import PolynomialFeatures
    data_modifier = PolynomialFeatures(degree=deg)
    in_data_poly = data_modifier.fit_transform(in_data)
    return in_data_poly

def Pred(test_data, model):
    y_pred = model.predict(test_data)
    return y_pred


def visualize(in_data, out_data, model):
    plt.scatter(in_data, out_data, color='red')
    plt.plot(in_data, Pred(in_data, model), color='blue')
    plt.title('Truth or Bluff')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


if __name__ == "__main__":
    dataset = read_file("Position_Salaries")
    X, y = get_InOutparams(dataset)

    #Ignoring the Position Parameter in this dataset
    X = X[:, 1:]

    #Fitting the linear regression model
    regressor_lin = LinReg(X, y)

    #Visualising the linear model
    visualize(X, y, regressor_lin)

    #Fitting the polynomial regression model
    X_poly = PolynomialFeatures(X, deg=2)
    regressor_poly = LinReg(X_poly, y)

    #visualizing the polynomial regression model
    plt.scatter(X, y, color='red')
    plt.plot(X, regressor_poly.predict(X_poly), color='blue')
    plt.title('Truth or Bluff')
    plt.show()