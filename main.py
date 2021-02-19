import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

def KFoldSplit(X, y):
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

    df.columns = ['A' + str((i + 1)) for i in range(df.shape[1])]

    # remove missing values
    for i in df.columns:
        df = df.drop(index=df[df[i].astype(str) =='?'].index, axis=0)

    # shuffle
    df = shuffle(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

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
    
    # normalise
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # encoding y
    y = y.astype("str")
    y.loc[y == '+'] = '1'
    y.loc[y == '-'] = '0'
    y = np.array(y.astype('int'))

    return X, y

def normal_train(X, y):
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy, balanced = eval(y_test, y_pred)

def KFold_train(X, y):
    accuracies = []
    balanced_accuracies = []

    for X_train, X_test, y_train, y_test in KFoldSplit(X, y):
        clf = LogisticRegression()
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        accuracy, balanced = eval(y_test, y_pred)
        accuracies.append(accuracy)
        balanced_accuracies.append(balanced)
    print('Avg Accuracy: ', round(np.mean(accuracies), 3))
    print('Avg Balanced Accuracy: ', round(np.mean(balanced_accuracies), 3))


def eval(y_test, y_pred):
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    balanced_accu = round(balanced_accuracy_score(y_test, y_pred), 3)
    
    print('accuracy: ', accuracy)
    print('balanced accuracy: ', balanced_accu)

    return accuracy, balanced_accu


if __name__ == "__main__":
    X, y = preprocess("data/crx.data")
    normal_train(X, y)


