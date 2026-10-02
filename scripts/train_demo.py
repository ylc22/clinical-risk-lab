from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from clinical_risk_lab.modeling import make_pipeline, choose_threshold, classification_metrics
rng=np.random.default_rng(7); n=6000
age=rng.normal(55,15,n).clip(18,90); bmi=rng.normal(28,5,n).clip(16,50); sbp=rng.normal(128,18,n).clip(85,210); a1c=rng.normal(5.9,1.2,n).clip(4,14); hdl=rng.normal(50,13,n).clip(15,100); smoker=rng.choice(["yes","no"],n,p=[.23,.77]); sex=rng.choice(["F","M"],n)
logit=-7.3+.045*age+.06*(bmi-25)+.018*(sbp-120)+.5*(a1c-5.5)-.018*(hdl-50)+.8*(smoker=="yes")+.35*((age>65)&(a1c>6.5))
p=1/(1+np.exp(-logit)); y=rng.binomial(1,p)
X=pd.DataFrame(dict(age=age,bmi=bmi,sbp=sbp,a1c=a1c,hdl=hdl,smoker=smoker,sex=sex))
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.4,random_state=42,stratify=y); Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42,stratify=ytmp)
base=make_pipeline().fit(Xtr,ytr); cal=CalibratedClassifierCV(base,method="isotonic",cv=5).fit(Xtr,ytr)
pv=cal.predict_proba(Xv)[:,1]; t=choose_threshold(yv,pv); pt=cal.predict_proba(Xte)[:,1]
out=ROOT/"outputs"; out.mkdir(exist_ok=True); (ROOT/"data").mkdir(exist_ok=True)
metrics=classification_metrics(yte,pt,t)
for group in ["sex","smoker"]:
    metrics[f"subgroup_{group}"]={}
    for level in sorted(Xte[group].unique()):
        m=Xte[group].values==level
        metrics[f"subgroup_{group}"][level]=classification_metrics(yte[m],pt[m],t)
with open(out/"metrics.json","w") as f: json.dump(metrics,f,indent=2)
joblib.dump({"model":cal,"threshold":t},out/"model.joblib")
prob_true,prob_pred=calibration_curve(yte,pt,n_bins=10,strategy="quantile")
fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],ls="--"); ax.plot(prob_pred,prob_true,marker="o"); ax.set(xlabel="Predicted risk",ylabel="Observed event rate",title="Calibration curve"); fig.tight_layout(); fig.savefig(out/"calibration.png",dpi=160); plt.close(fig)
perm=permutation_importance(cal,Xte,yte,n_repeats=5,random_state=1,scoring="roc_auc")
imp=pd.DataFrame({"feature":X.columns,"importance":perm.importances_mean}).sort_values("importance",ascending=False); imp.to_csv(out/"feature_importance.csv",index=False)
print(json.dumps(metrics,indent=2)); print("\nTop features:\n",imp.head().to_string(index=False))
