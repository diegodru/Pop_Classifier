from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import json
import sys
import pickle as pk

def main():
    tagsFile = open(sys.argv[2])
    tags = json.load(tagsFile)
    df = pd.read_csv(sys.argv[1], header=None)
    outfile = open(sys.argv[3], "wb")
    X = pd.DataFrame(df.drop(0, axis=1)).to_numpy()
    y = np.ravel(pd.DataFrame(df[0]).to_numpy())
    for i in range(len(y)):
        y[i] = tags[y[i]]['marca']
    nn = neighbors.KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=15, max_features=25, n_jobs=5)
    algo = rf
    algo.fit(X, y)
    y_train = algo.predict(X)
    print(accuracy_score(y_train, y))
    print(f1_score(y_train, y, average="macro"))
    pk.dump(obj=algo, file=outfile)

if __name__ == "__main__":
    main()
