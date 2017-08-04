import pandas as pd
import matplotlib.pyplot as plt
import numpy

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
    SS = StandardScaler();
    array = SS.fit_transform(array)
    return array

def SVR(in_data, out_data):
    from sklearn.svm import SVR
    regressor = SVR(kernel='rbf')
    regressor.fit(in_data, out_data)
    return regressor

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

    #Need to scale data for SVR
    X = FeatureScaler(X)
    y = FeatureScaler(y)

    #Getting the regression model
    SVRegressor = SVR(X, y)
    visualize(X, y, SVRegressor)

    #An example showing a prediction when there is scaling:





