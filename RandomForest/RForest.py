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

def RandomForestReg(in_data, out_data):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=300) #n_estimators is the number of trees
    regressor.fit(in_data, out_data)
    return regressor

def visualizeHighRes(in_data, out_data, model):
    X_grid = np.arange(min(in_data), max(in_data), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(in_data, out_data, color='red')
    plt.plot(X_grid, model.predict(X_grid), color='blue')
    plt.title('Truth or Bluff')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

if __name__ == "__main__":
    dataset = read_file("Position_Salaries")
    X, y = get_InOutparams(dataset)
    X = X[:, 1:]
    RForestReg = RandomForestReg(X, y)
    #visualizeHighRes(X, y, RForestReg)
    print(RForestReg.predict(6.5))
