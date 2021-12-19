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
    df = pd.read_csv(sys.argv[1], header=None)
    in_clasificador = open(sys.argv[2], "rb")
    clasificador = pk.load(in_clasificador)
    X = df.drop(0, axis=1)
    X = pd.DataFrame(X).to_numpy()
    y = df[0]
    y = np.ravel(pd.DataFrame(y).to_numpy())
    y_pred = clasificador.predict(X)
    #np.savetxt(outfile, y_pred, "%s", ",")
    out = dict()
    for i in range(len(y_pred)):
        out[y[i]] = { "marca": y_pred[i] }
    outfile = open(sys.argv[3], "w")
    json.dump(out, outfile, indent=4, sort_keys=True)
    


if __name__ == "__main__":
    main()
