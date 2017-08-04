import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    y_temp = y;

    #Ignoring the Position Parameter in this dataset
    X = X[:, 1:]
    x_temp = X;

    #Need to scale data for SVR
    X = FeatureScaler(X)
    y = FeatureScaler(y)

    #Getting the regression model
    SVRegressor = SVR(X, y)
    visualize(X, y, SVRegressor)

    #An example showing a prediction when there is scaling:
    from sklearn.preprocessing import StandardScaler
    SS_x = StandardScaler()
    SS_y = StandardScaler()

    SS_x.fit_transform(x_temp)
    SS_y.fit_transform(y_temp)

    x_scaled = SS_x.transform(np.array([[6.5]]))

    y_pred = SVRegressor.predict(6.5)
    y_pred_1 = SS_y.inverse_transform(y_pred)

    y_pred_scaled = SVRegressor.predict(x_scaled)
    y_pred_scaled_1 = SS_y.inverse_transform(y_pred_scaled)

    print("Without Scaling 6.5, model says: {0}\n".format(y_pred))
    print("Without Scaling 6.5, the inverse transformation on the model's output is: {0}\n".format(y_pred_1))

    print("Scaling 6.5, model says: {0}\n".format(y_pred_scaled))
    print("Scaling 6.5, the inverse transformation on the model's output is: {0}\n".format(y_pred_scaled_1))






