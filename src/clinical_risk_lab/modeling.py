import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

NUM = ["age","bmi","sbp","a1c","hdl"]
CAT = ["smoker","sex"]


def make_pipeline():
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def choose_threshold(y, p):
    grid = np.linspace(.05,.95,181)
    best=(.5,-9)
    for t in grid:
        pred=(p>=t).astype(int)
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
        j=sens+spec-1
        if j>best[1]: best=(float(t),float(j))
    return best[0]


def classification_metrics(y,p,t=.5):
    pred=(p>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        "auroc": roc_auc_score(y,p),
        "auprc": average_precision_score(y,p),
        "brier": brier_score_loss(y,p),
        "sensitivity": tp/max(tp+fn,1),
        "specificity": tn/max(tn+fp,1),
        "threshold": t,
    }
