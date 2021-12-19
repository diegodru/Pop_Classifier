from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn import neighbors
import pandas as pd
import numpy as np
import json
import sys
import pickle as pk

def main():
    if not len(sys.argv) == 4:
        print(f"usage {sys.argv[0]} <in_features> <in_clasificador> <out_predicciones_write_file>", file=sys.stderr)
        exit(1)
    df = pd.read_csv(sys.argv[1], header=None)
    in_clasificador = open(sys.argv[2], "rb")
    clasificador = pk.load(in_clasificador)
    X = pd.DataFrame(df.drop(0, axis=1)).to_numpy()
    y = np.ravel(pd.DataFrame(df[0]).to_numpy())
    y_pred = clasificador.predict(X)
    #np.savetxt(outfile, y_pred, "%s", ",")
    out = dict()
    for i in range(len(y_pred)):
        out[y[i]] = { "marca": y_pred[i] }
    outfile = open(sys.argv[3], "w")
    json.dump(out, outfile, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
