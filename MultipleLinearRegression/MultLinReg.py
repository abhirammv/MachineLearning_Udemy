#Implementation of Multiple Linear Regression
#The procedure is very similar to Simple linear regression, except that you have multiple predictors
#Preprocess the data - Missing value handling, outlier handling, feature scaling and obtaining the train/test data split

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

#Encoding categorical data
def EncodeCatData(array, type = "indep"):
    from sklearn.preprocessing import LabelEncoder
    LE = LabelEncoder()

    if(type == "dep"):
        array = LE.fit_transform(array)
        return array
    else:
        array[:, 3] = LE.fit_transform(array[:, 3])

    from sklearn.preprocessing import OneHotEncoder
    OHE = OneHotEncoder(categorical_features=[3]);
    array = OHE.fit_transform(array).toarray()
    return array

def TrainTestSplit(array):
    from sklearn.cross_validation import train_test_split
    array_tr, array_te = train_test_split(array, test_size=0.2, random_state=0)
    return array_tr, array_te

def LinReg(in_data, out_data):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(in_data, out_data)
    return regressor

def LinPred(test_set, lin_model):
    from sklearn.linear_model import LinearRegression
    y_pred = lin_model.predict(test_set)
    return y_pred

if __name__ == "__main__":
    dataset = read_file("50_Startups")
    X, y = get_InOutparams(dataset)
    X = EncodeCatData(X)
    #Avoiding the dummy variable trap
    X = X[:, 1:]
    X_tr, X_te = TrainTestSplit(X)
    y_tr, y_te = TrainTestSplit(y)
    regressor_fit = LinReg(X_tr, y_tr)
    y_pred = LinPred(X_te, regressor_fit)

    #Backward elimintation process
    import statsmodels.formula.api as sm

    #Setting the significance level of 5%

    #Adding a column of ones to the act as the intercept
    X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

    X_opt = X[:, [0, 3]]
    print(sm.OLS(endog=y, exog=X_opt).fit().summary())












