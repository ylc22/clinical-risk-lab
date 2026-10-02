import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from clinical_risk_lab.modeling import choose_threshold, classification_metrics


def test_threshold_range():
    t=choose_threshold(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))
    assert 0<t<1


def test_metrics_keys():
    m=classification_metrics(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]),.5)
    assert {"auroc","auprc","brier","sensitivity","specificity","threshold"}<=set(m)
