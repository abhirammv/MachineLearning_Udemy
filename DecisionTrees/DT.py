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

def DTReg(in_data, out_data):
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor()
    regressor.fit(in_data, out_data)
    return regressor

def Pred(test_data, model):
    pred = model.predict(test_data)
    return pred

def visualize(in_data, out_data, model):
    plt.scatter(in_data, out_data, color='red')
    plt.plot(in_data, Pred(in_data, model), color='blue')
    plt.title('Truth or Bluff')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

def visualizeHighRes(in_data, out_data, model):
    X_grid = np.arange(min(in_data), max(in_data), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(in_data, out_data, color='red')
    plt.plot(X_grid, Pred(X_grid, model), color='blue')
    plt.title('Truth or Bluff')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

if __name__ == "__main__":
    dataset = read_file("Position_Salaries")
    X, y = get_InOutparams(dataset)
    X = X[:, 1:]
    DTRegressor = DTReg(X, y)
    visualize(X, y, DTRegressor)
    visualizeHighRes(X, y, DTRegressor)