from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("../../data/processed/binary_dataset.parquet")

LEAKY_COLS = []
# LEAKY_COLS = ["f_gap_open"]

drop_cols = ["datetime", "symbol"] + LEAKY_COLS
features = [c for c in df.columns if c.startswith("f_")]
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
y = df["label"].astype(int)
dates = pd.to_datetime(df["datetime"])

unique_days = np.sort(dates.unique())
test_days = unique_days[-40:] if len(unique_days) > 60 else unique_days[int(len(unique_days)*0.2):]
is_test = dates.isin(test_days)
X_train, X_test = X[~is_test], X[is_test]
y_train, y_test = y[~is_test], y[is_test]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}, "
      f"Train days: {X.loc[~is_test, :].shape[0]//y[~is_test].groupby(dates[~is_test]).size().mean():.0f} avg/day")

clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight=None  # 若样本不均衡明显，可设为 "balanced"
)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

acc = accuracy_score(y_test, pred)
try:
    auc = roc_auc_score(y_test, proba)
except ValueError:
    auc = float("nan")

print("\n=== Metrics (Time-split) ===")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC : {auc:.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, pred, digits=4))


imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== Top-20 Feature Importances ===")
print(imp.head(20))
