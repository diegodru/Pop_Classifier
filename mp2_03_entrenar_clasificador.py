from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import json
import sys
import pickle as pk

def main():
    np.set_printoptions(threshold=sys.maxsize) 
    if not len(sys.argv) == 4:
        print(f"usage {sys.argv[0]} <in_features> <in_etiquetas> <out_classifier>", file=sys.stderr)
        exit(1)
    tagsFile = open(sys.argv[2])
    tags = json.load(tagsFile)
    df = pd.read_csv(sys.argv[1], header=None)
    outfile = open(sys.argv[3], "wb")
    X = pd.DataFrame(df.drop(0, axis=1)).to_numpy()
    y = np.ravel(pd.DataFrame(df[0]).to_numpy())
    for i in range(len(y)):
        y[i] = tags[y[i]]['marca']
    nn = neighbors.KNeighborsClassifier(n_neighbors=5)
    crit, trees, max_d, max_feat = "gini", 100, 20, 25
    rf = RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=15, max_features=25, n_jobs=5)
    rf = RandomForestClassifier(n_estimators=trees, criterion=crit, max_depth=max_d, max_features=max_feat, n_jobs=5)
    algo = rf
    algo.fit(X, y)
    y_train = algo.predict(X)
    print(confusion_matrix(y_train, y))
    print(accuracy_score(y_train, y))
    print(f1_score(y_train, y, average="macro"))
    pk.dump(obj=algo, file=outfile)

if __name__ == "__main__":
    main()
