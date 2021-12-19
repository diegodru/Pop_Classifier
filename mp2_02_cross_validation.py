from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn import neighbors
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import json
import sys
import pickle as pk

def main():
    tagsFile = open(sys.argv[2])
    tags = json.load(tagsFile)
    df = pd.read_csv(sys.argv[1], header=None)
    X = df.drop(0, axis=1)
    X = pd.DataFrame(X).to_numpy()
    y = df[0]
    y = pd.DataFrame(y).to_numpy()
    y = np.ravel(y)
    for i in range(len(y)):
        y[i] = tags[y[i]]['marca']
    kfold = KFold(n_splits=5, shuffle=True)
    nn = neighbors.KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=5, max_features=10)
    algo = nn
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)
        print(f1_score(y_test, y_pred, average="micro"))
        #print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
