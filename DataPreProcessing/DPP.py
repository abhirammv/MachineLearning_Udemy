#The template for data preprocessing

#Libraries used
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import scipy

#The function to read a csv file using pandas. Returns an array of values;
#Return type is pandas.DataFrame;
#To index data, as rows and columns, use <dataset_name>.iloc[rows-range, columns-range];

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    return dataset

#A function to get input and output attributes seperately.
#The independent data is referred to as X, and the dependant data is referred to as y.
#Main assumption: The last column represents the dependant variable
#The return types are numpy.ndarray. Technically it's a list of lists in case of multiple columns
#Basically, it's the list that contain the entries of each row as elements

def get_InOutparams(data):

    #Input:
    X = data.iloc[:, :-1].values

    #Output
    y = data.iloc[:, -1].values

    return X, y

#Handling missing data - optional
def RemoveMissingValues(array):
    from sklearn.preprocessing import Imputer #imports the imputer class in sklearn.preprocessing library

    imputer_obj = Imputer(missing_values="NaN", strategy="mean", axis=0)

    #This creates an imputer obj, that handles missing values when fit to an array
    #The missing values are represented by "NaN", the strategy to replace them is averaging along the columns
    #axis = 0 asks it to impute along the columns, means to go through rows of each column and impute

    NumColumns = len(array[0])

    #Impute along all the rows of each column, for all the columns without strings in them
    #Imputer doesn't work with strings
    #Assumes the 1st column, indexed by 0, has the column of strings

    return imputer_obj.fit_transform(array[:, 1:NumColumns])

#Encoding Categorical Data
def EncodeCatData(array, type = "indep"):
    #A Label encoder encodes the different types into integers
    #If there are 3 categories named A,B and C then A -> 0, B -> 1 and C -> 2
    #After encoding, they have to be converted into dummy variables
    #Else the linear model will think of them as actual values and a higher priority might be given
    # to the category with the largest representative value.
    #If the type is independant, only label encoding is sufficient

    #Assumes that the first column has categorical variables

    from sklearn.preprocessing import LabelEncoder
    LE = LabelEncoder()

    if(type == "dep"):
        array = LE.fit_transform(array)
        return array
    else:
        array[:, 0] = LE.fit_transform(array[:, 0])


    from sklearn.preprocessing import OneHotEncoder
    OHE = OneHotEncoder(categorical_features=[0]);
    array = OHE.fit_transform(array).toarray()
    return array

#Feature Scaling
def FeatureScaler(array):
    from sklearn.preprocessing import StandardScaler
    SS = StandardScaler();
    array = SS.fit_transform(array)
    return array

#Splitting the data into train/test data
def TrainTestSplit(array):
    from sklearn.cross_validation import train_test_split
    array_tr, array_te = train_test_split(array, test_size=0.2, random_state=0)
    return array_tr, array_te



if __name__ == "__main__":

    #Function call to read the file and get raw data
    dataset = read_file("Data")
    X, y = get_InOutparams(dataset)
    X[:, 1:3] = RemoveMissingValues(X)
    X = EncodeCatData(X)
    y = EncodeCatData(y, type="dep")
    X = FeatureScaler(X)
    X_tr, X_te = TrainTestSplit(X)
    y_tr, y_te = TrainTestSplit(y)