import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

# load data
df = pd.read_csv('data/crx.data',sep=',', header=None)

df.columns = ['A' + str((i + 1)) for i in range(df.shape[1])]

# remove missing values
for i in df.columns:
    df = df.drop(index=df[df[i]=='?'].index, axis=0)

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

# encoding y
y.loc[y == '+'] = 1
y.loc[y == '-'] = 0

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('succeed')

clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


print('accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('balanced accuracy: ' + str(balanced_accuracy_score(y_test, y_pred)))