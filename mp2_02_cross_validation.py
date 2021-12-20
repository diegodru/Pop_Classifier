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

def crossValidate(kfold, algo, X, y):
    f1 = None
    acc = None
    conf_mat = {}
    for clase in y:
        if not clase in conf_mat:
            d = dict()
            for key in y:
                d[key] = 0
            conf_mat[clase] = d
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)
        if f1 is None:
            f1 = np.array(f1_score(y_test, y_pred, average="macro"))
        else:
            f1 = np.append(f1, [f1_score(y_test, y_pred, average="macro")])
        if acc is None:
            acc = np.array(accuracy_score(y_test, y_pred))
        else:
            acc = np.append(acc, [accuracy_score(y_test, y_pred)])
        for i in range(len(y_test)):
            conf_mat[y_test[i]][y_pred[i]] += 1
    f1 = np.append(f1, [np.average(f1)])
    acc = np.append(acc, [np.average(acc)])
    return (f1, acc, conf_mat)

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
    nn = neighbors.KNeighborsClassifier(n_neighbors=5)
    kfold = KFold(n_splits=int(sys.argv[3]), shuffle=True)
    crit, trees, max_d, max_feat = "gini", 100, 20, 25
    rf = RandomForestClassifier(n_estimators=trees, criterion=crit, max_depth=max_d, max_features=max_feat, n_jobs=5)
    f1_scores, acc, conf_matrix = crossValidate(kfold, rf, X, y)
    avg_f1_por_clase = None
    array = []
    for clase in conf_matrix:
        fn, fp, rec, prec, f1 = 0, 0, 0, 0, 0
        arr = []
        for key in conf_matrix[clase]:
            if not key == clase:
                fp += conf_matrix[clase][key]
                fn += conf_matrix[key][clase]
            arr.append(conf_matrix[clase][key])
        rec = conf_matrix[clase][clase] / (conf_matrix[clase][clase] + fn)
        prec = conf_matrix[clase][clase] / (conf_matrix[clase][clase] + fp)
        f1 = 2 * prec * rec / (rec + prec)
        if avg_f1_por_clase is None:
            avg_f1_por_clase = np.array(f1)
        else:
            avg_f1_por_clase = np.append(avg_f1_por_clase, [f1])
        array.append(arr)
    matriz_confusion = np.array(array)
    np.savetxt("matriz_confusion.csv", matriz_confusion, "%i", ",")
    print("f1 por clase promedio: %f" % (np.average(avg_f1_por_clase)), file=sys.stderr)
    print("P1, P2, P3, P4, P5, Promedio")
    print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % tuple(f1_scores))
    print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % tuple(acc))
                    




if __name__ == "__main__":
    main()
