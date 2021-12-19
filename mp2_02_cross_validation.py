from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn import neighbors
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import sklearn
import pandas as pd
import numpy as np
import json
import sys
import pickle as pk

def main():
    if not len(sys.argv) == 4:
        print(f"usage {sys.argv[0]} <in_features> <in_etiquetas> <n_folds>", file=sys.stderr)
        exit(1)
    tagsFile = open(sys.argv[2])
    tags = json.load(tagsFile)
    df = pd.read_csv(sys.argv[1], header=None)
    X = pd.DataFrame(df.drop(0, axis=1)).to_numpy()
    y = np.ravel(pd.DataFrame(df[0]).to_numpy())
    for i in range(len(y)):
        y[i] = tags[y[i]]['marca']
    kfold = KFold(n_splits=int(sys.argv[3]), shuffle=True)
    nn = neighbors.KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=15, max_features=25, n_jobs=5)
    algo = rf
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)
        #print(accuracy_score(y_test, y_pred))
        print(f1_score(y_test, y_pred, average="macro"))
        #print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
