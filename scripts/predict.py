from pathlib import Path
import argparse, joblib, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
for x in ["age","bmi","sbp","a1c","hdl"]: p.add_argument(f"--{x}",type=float,required=(x!="hdl"),default=50)
p.add_argument("--smoker",choices=["yes","no"],required=True); p.add_argument("--sex",choices=["F","M"],required=True)
a=p.parse_args(); bundle=joblib.load(ROOT/"outputs"/"model.joblib")
X=pd.DataFrame([{k:getattr(a,k) for k in ["age","bmi","sbp","a1c","hdl","smoker","sex"]}]); risk=float(bundle["model"].predict_proba(X)[0,1]); print(f"Predicted event risk: {risk:.1%}"); print(f"Decision threshold: {bundle['threshold']:.2f}")
