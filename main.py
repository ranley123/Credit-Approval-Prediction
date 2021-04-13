import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def KFoldSplit(X, y):
    '''
    This function would do K fold split and return each fold
    '''
    kf = KFold()
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]

        yield X_train, X_test, y_train, y_test

def preprocess(filename, sep=','):
    # load data
    df = pd.read_csv(filename, sep=sep, header=None)

    # give each column a name
    df.columns = ['A' + str((i + 1)) for i in range(df.shape[1])]

    # remove missing values
    for i in df.columns:
        df = df.drop(index=df[df[i].astype(str) =='?'].index, axis=0)

    # shuffle before splitting
    df = shuffle(df)

    # get X and y respectively
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # those attributes are categorical, needed to be encoded
    indices = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

    for name in indices:
        one_hot = pd.get_dummies(X[name])
        names = []
        for col in one_hot.columns:
            names.append(name + '_' + col)
        # Drop column as it is now encoded
        X = X.drop(name,axis = 1)
        # Join the encoded df
        one_hot.columns = names
        X = X.join(one_hot)

    # encoding y
    y = y.astype("str")
    y.loc[y == '+'] = '1'
    y.loc[y == '-'] = '0'
    y = np.array(y.astype('int'))

    return X.to_numpy(), y

def train(X, y):
    all_y_true = []
    all_y_pred = []
    all_y_pred_balanced = []

    for X_train, X_test, y_train, y_test in KFoldSplit(X, y):
        # normalise
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        clf = LogisticRegression(penalty='none', class_weight=None)
        clf_balanced = LogisticRegression(penalty='none', class_weight='balanced')
        
        # fit the training set
        clf.fit(X_train,y_train)
        clf_balanced.fit(X_train,y_train)

        # get predicted values
        y_pred = clf.predict(X_test)
        y_pred_balanced = clf_balanced.predict(X_test)

        # store predicted values
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_pred_balanced.extend(y_pred_balanced)

        # plot for each fold
        # plot_precision_recall_curve(clf, X_test, y_test).figure_.show()
        # plot_precision_recall_curve(clf_balanced, X_test, y_test)
    
    plt.show()

    # evaluate 
    accuracy, balanced = eval(all_y_true, all_y_pred)
    accuracy_balanced, balanced_balancend = eval(all_y_true, all_y_pred_balanced)
    print('Normal LR Avg Accuracy: ', accuracy)   
    print('Normal LR Avg Balanced Accuracy: ', balanced)
    print('Balanced LR Avg Accuracy: ', accuracy_balanced)   
    print('Balanced LR Avg Balanced Accuracy: ', balanced_balancend)
    print('Normal Confusion Matrix: ', confusion_matrix(all_y_true, all_y_pred))     
    print('Balanced Confusion Matrix: ', confusion_matrix(all_y_true, all_y_pred_balanced)) 


def eval(y_test, y_pred):
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    balanced_accu = round(balanced_accuracy_score(y_test, y_pred), 3)

    return accuracy, balanced_accu


if __name__ == "__main__":
    X, y = preprocess("data/crx.data")
    train(X, y)


