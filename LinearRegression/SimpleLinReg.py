#Implementation of Simple Linear Regression
#Steps:
"""
1) Preprocess the data -> Can be done by reusing the functions in DPP.py
    1. Handle missing values
    2. Apply feature scaling if required **
    3. Split the data into training and test sets

2) Fit the linear regression model found in the sklearn package

** -> Some functions take have built-in feature scaling. Read the documentation once to be thorough
"""
import pandas as pd
import numpy
import matplotlib.pyplot as plt

#Implementation form DPP.py ---------------------
#------------------------------------------------

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    return dataset

def get_InOutparams(data):

    #Input:
    X = data.iloc[:, :-1].values

    #Output
    y = data.iloc[:, -1].values

    return X, y

def TrainTestSplit(array):
    from sklearn.cross_validation import train_test_split
    array_tr, array_te = train_test_split(array, test_size=0.2, random_state=0)
    return array_tr, array_te

#------------------------------------------------

#Uses the sklearn.linear_model to fit a linear regression model
#Inputs: The input and output datasets
#Returns the object of LinearRegression class that has fit the input and output data
def LinReg(in_data, out_data):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(in_data, out_data)
    return regressor

#To predict on the test set
#Uses the regressor that is fit using the LinReg() function

def LinPred(test_set, lin_model):
    from sklearn.linear_model import LinearRegression
    y_pred = lin_model.predict(test_set)
    return y_pred

#Visualize the predictions by creating a scatter plot
def visualize(in_data, out_data, model, train_data):
    plt.scatter(in_data, out_data, color='red')
    plt.plot(in_data, LinPred(train_data, model), color='blue')
    plt.title('Salary vs Experience (Training set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()


if __name__ == "__main__":
    dataset = read_file("Salary_Data")
    X, y = get_InOutparams(dataset)
    X_tr, X_te = TrainTestSplit(X)
    y_tr, y_te = TrainTestSplit(y)
    regressor_fit = LinReg(X_tr, y_tr)
    predictions = LinPred(X_te, regressor_fit)
    visualize(X_tr, y_tr, regressor_fit, X_tr)
    visualize(X_te, y_te, regressor_fit, X_te)




