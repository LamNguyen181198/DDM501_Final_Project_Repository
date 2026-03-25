"""Quick smoke-test for the Responsible AI notebook dependencies."""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'pipeline')

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for test
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ARTS = Path('artifacts')

# --- Load data ---
train_df = pd.read_parquet(ARTS / 'train_split.parquet')
test_df  = pd.read_parquet(ARTS / 'test_split.parquet')
X_tr = np.load(ARTS / 'X_train.npy')
y_tr = np.load(ARTS / 'y_train.npy')
X_te = np.load(ARTS / 'X_test.npy')
y_te = np.load(ARTS / 'y_test.npy')
le   = joblib.load(ARTS / 'label_encoder.joblib')
pre  = joblib.load(ARTS / 'preprocessor.joblib')
print(f"Data loaded: X_train={X_tr.shape}  X_test={X_te.shape}")
print(f"Classes: {le.classes_}")
print(f"Test columns: {list(test_df.columns[:8])}")

# --- Train RF ---
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
acc = accuracy_score(y_te, rf.predict(X_te))
print(f"RF Test Accuracy: {acc:.4f}")

# --- Feature names ---
num_features  = pre.transformers_[0][2]
ord_features  = pre.transformers_[1][2]
nom_encoder   = pre.transformers_[2][1]
nom_base      = pre.transformers_[2][2]
nom_features  = list(nom_encoder.get_feature_names_out(nom_base))
FEATURE_NAMES = list(num_features) + list(ord_features) + nom_features
print(f"Total features: {len(FEATURE_NAMES)}")

# --- SHAP ---
bg  = X_tr[np.random.default_rng(42).choice(len(X_tr), 100, replace=False)]
exp = shap.TreeExplainer(rf, data=bg, feature_names=FEATURE_NAMES)
sv  = exp(X_te[:30])
sat_idx = list(le.classes_).index('Satisfied')
sv_sat  = sv[:, :, sat_idx]
mean_abs = pd.Series(np.abs(sv_sat.values).mean(axis=0), index=FEATURE_NAMES)
print(f"SHAP values shape: {sv_sat.values.shape}  OK")
print("Top 5 features by mean |SHAP|:")
print(mean_abs.sort_values(ascending=False).head(5).to_string())

# --- Fairness quick check ---
preds = le.inverse_transform(rf.predict(X_te))
true  = le.inverse_transform(y_te)
test_copy = test_df.copy().reset_index(drop=True)
test_copy['y_pred'] = preds
test_copy['y_true'] = true
test_copy['correct'] = (test_copy['y_pred'] == test_copy['y_true']).astype(int)
by_gender = test_copy.groupby('gender')['correct'].mean()
print(f"\nAccuracy by gender:\n{by_gender}")
by_age = test_copy.groupby('age_group')['correct'].mean()
print(f"\nAccuracy by age_group:\n{by_age}")

print("\n=== All smoke tests PASSED ✅ ===")
