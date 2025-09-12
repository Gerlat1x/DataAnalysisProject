from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# LightGBM / XGBoost
import lightgbm as lgb
import xgboost as xgb

# ===== 1) 读数据 =====
df = pd.read_parquet("../../data/processed/binary_dataset.parquet")

# 只用 f_ 开头的特征，避免泄露
features = [c for c in df.columns if c.startswith("f_")]
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
y = df["label"].astype(int)
dates = pd.to_datetime(df["datetime"])

# ===== 2) 时间切分 =====
unique_days = np.sort(dates.unique())
test_days = unique_days[-40:] if len(unique_days) > 60 else unique_days[int(len(unique_days)*0.2):]
is_test = dates.isin(test_days)

X_train, X_test = X[~is_test], X[is_test]
y_train, y_test = y[~is_test], y[is_test]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ===== 3) LightGBM =====
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params_lgb = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "is_unbalance": True
}

model_lgb = lgb.train(
    params_lgb,
    lgb_train,
    valid_sets=[lgb_train, lgb_test],
    num_boost_round=500,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),
    ],
)

proba_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
pred_lgb = (proba_lgb >= 0.5).astype(int)

print("\n=== LightGBM Results ===")
print("Accuracy:", accuracy_score(y_test, pred_lgb))
print("ROC-AUC :", roc_auc_score(y_test, proba_lgb))
print(classification_report(y_test, pred_lgb, digits=4, zero_division=0))

# ===== 4) XGBoost =====
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params_xgb = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",   # 或 "gpu_hist" 如果你装了 GPU 版
    "nthread": -1,
    "seed": 42,
}

model_xgb = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=50,
    verbose_eval=50,
)

proba_xgb = model_xgb.predict(dtest, iteration_range=(0, model_xgb.best_iteration))
pred_xgb = (proba_xgb >= 0.5).astype(int)

print("\n=== XGBoost Results ===")
print("Accuracy:", accuracy_score(y_test, pred_xgb))
print("ROC-AUC :", roc_auc_score(y_test, proba_xgb))
print(classification_report(y_test, pred_xgb, digits=4))

# prec, rec, thr = precision_recall_curve(y_test, proba_lgb)
# plt.plot(rec, prec)
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve (LightGBM)")
# plt.show()
