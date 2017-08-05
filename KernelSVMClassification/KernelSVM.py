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

def FeatureScaler(array):
    from sklearn.preprocessing import StandardScaler
    SS = StandardScaler()
    array = SS.fit_transform(array)
    return array

def TrainTestSplit(array):
    from sklearn.cross_validation import train_test_split
    array_tr, array_te = train_test_split(array, test_size=0.2, random_state=0)
    return array_tr, array_te

def KernelSVM(in_data, out_data):
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf')
    classifier.fit(in_data, out_data)
    return classifier

def Pred(test_data, model):
    pred = model.predict(test_data)
    return pred

def GetConfMatrix(actual_values, pred_values):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual_values, pred_values)
    return cm

def visualize(in_data, out_data, model, mode):
    from matplotlib.colors import ListedColormap
    X_set, y_set = in_data, out_data
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    if(mode=="training"):
        plt.title('Kernel SVM Classification (Training set)')
    elif(mode=="test"):
        plt.title('Kernel SVM Classification (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    dataset = read_file("Social_Network_Ads")
    X, y = get_InOutparams(dataset)
    X = X[:, 2:4] #Selection of parameters
    X = FeatureScaler(X)
    X_tr, X_te = TrainTestSplit(X)
    y_tr, y_te = TrainTestSplit(y)
    KernelSVMClassifier = KernelSVM(X_tr, y_tr)
    y_pred = Pred(X_te, KernelSVMClassifier)
    ConfMtx = GetConfMatrix(y_te, y_pred)
    print(ConfMtx)
    visualize(X_tr, y_tr, KernelSVMClassifier, "training")
    visualize(X_te, y_te, KernelSVMClassifier, "test")
