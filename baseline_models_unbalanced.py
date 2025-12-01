import numpy as np
import pandas as pd
import optuna
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_OUTER = 3
N_INNER = 3

OUT_DIR = "baseline_results_unbalanced"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
df = pd.read_csv("SCF2022_fragility.csv")

# X: predictors, y: target
X = df.drop(columns=["fragile", "YY1"])
y = df["fragile"]

continuous_cols = [
    c for c in X.columns
    if not set(X[c].dropna().unique()).issubset({0, 1})
]

outer_cv = StratifiedKFold(
    n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE
)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
    }

def save_confusion_heatmap(cm, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def objective_logit(trial, X_train, y_train, cv_inner):
    C = trial.suggest_float("C", 1e-4, 1e4, log=True)

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    return cross_val_score(
        model, X_train, y_train,
        cv=cv_inner,
        scoring="roc_auc"
    ).mean()


def objective_rf(trial, X_train, y_train, cv_inner):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    }

    model = RandomForestClassifier(**params)

    return cross_val_score(
        model, X_train, y_train,
        cv=cv_inner,
        scoring="roc_auc"
    ).mean()


def objective_xgb(trial, X_train, y_train, cv_inner):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "random_state": RANDOM_STATE
    }

    model = XGBClassifier(**params)

    return cross_val_score(
        model, X_train, y_train,
        cv=cv_inner,
        scoring="roc_auc"
    ).mean()

# -------------------------------------------------------
# Nested Cross-Validation
# -------------------------------------------------------
outer_metrics = {"Logistic Regression": [], "Random Forest": [], "XGBoost": []}

outer_conf_matrices = {m: [] for m in outer_metrics}
all_y_true = {m: [] for m in outer_metrics}
all_y_pred = {m: [] for m in outer_metrics}
all_y_prob = {m: [] for m in outer_metrics}

best_hyperparams_per_fold = {m: [] for m in outer_metrics}

slug_map = {
    "Logistic Regression": "logit",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}

fold_idx = 0
for train_idx, test_idx in outer_cv.split(X, y):
    fold_idx += 1
    print(f"\n================ OUTER FOLD {fold_idx} / {N_OUTER} ================")

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_tr_scaled = X_tr.copy()
    X_te_scaled = X_te.copy()
    X_tr_scaled[continuous_cols] = scaler.fit_transform(X_tr[continuous_cols])
    X_te_scaled[continuous_cols] = scaler.transform(X_te[continuous_cols])

    cv_inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)

    # ---------- Logistic Regression ----------
    print("\n[Fold {}] Optuna: Logistic Regression".format(fold_idx))
    study_logit = optuna.create_study(direction="maximize")
    study_logit.optimize(
        lambda trial: objective_logit(trial, X_tr_scaled, y_tr, cv_inner),
        n_trials=30,
        show_progress_bar=False
    )

    logit_params = study_logit.best_params.copy()
    logit_params["fold"] = fold_idx
    best_hyperparams_per_fold["Logistic Regression"].append(logit_params)

    logit = LogisticRegression(
        C=study_logit.best_params["C"],
        max_iter=2000,
        solver="lbfgs",
        n_jobs=-1
    )
    logit.fit(X_tr_scaled, y_tr)
    y_pred = logit.predict(X_te_scaled)
    y_prob = logit.predict_proba(X_te_scaled)[:, 1]

    outer_metrics["Logistic Regression"].append(compute_metrics(y_te, y_pred, y_prob))
    all_y_true["Logistic Regression"].append(y_te.values)
    all_y_pred["Logistic Regression"].append(y_pred)
    all_y_prob["Logistic Regression"].append(y_prob)

    # ---------- Random Forest ----------
    print("\n[Fold {}] Optuna: Random Forest".format(fold_idx))
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(
        lambda trial: objective_rf(trial, X_tr_scaled, y_tr, cv_inner),
        n_trials=30,
        show_progress_bar=False
    )
    rf_params = study_rf.best_params.copy()
    rf_params["fold"] = fold_idx
    best_hyperparams_per_fold["Random Forest"].append(rf_params)

    rf = RandomForestClassifier(
        **study_rf.best_params,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_tr_scaled, y_tr)
    y_pred = rf.predict(X_te_scaled)
    y_prob = rf.predict_proba(X_te_scaled)[:, 1]

    outer_metrics["Random Forest"].append(compute_metrics(y_te, y_pred, y_prob))
    all_y_true["Random Forest"].append(y_te.values)
    all_y_pred["Random Forest"].append(y_pred)
    all_y_prob["Random Forest"].append(y_prob)

    # ---------- XGBoost ----------
    print("\n[Fold {}] Optuna: XGBoost".format(fold_idx))
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_tr_scaled, y_tr, cv_inner),
        n_trials=30,
        show_progress_bar=False
    )

    xgb_params = study_xgb.best_params.copy()
    xgb_params["fold"] = fold_idx
    best_hyperparams_per_fold["XGBoost"].append(xgb_params)

    xgb = XGBClassifier(
        **study_xgb.best_params,
        eval_metric="auc",
        objective="binary:logistic",
        tree_method="hist",
        random_state=RANDOM_STATE
    )
    xgb.fit(X_tr_scaled, y_tr)
    y_pred = xgb.predict(X_te_scaled)
    y_prob = xgb.predict_proba(X_te_scaled)[:, 1]

    outer_metrics["XGBoost"].append(compute_metrics(y_te, y_pred, y_prob))
    all_y_true["XGBoost"].append(y_te.values)
    all_y_pred["XGBoost"].append(y_pred)
    all_y_prob["XGBoost"].append(y_prob)

# -------------------------------------------------------
# Results
# -------------------------------------------------------
summary = {}
for model_name, folds in outer_metrics.items():
    keys = folds[0].keys()
    summary[model_name] = {k: np.mean([fold[k] for fold in folds]) for k in keys}

results_df = pd.DataFrame(summary).T
results_df.to_csv(f"{OUT_DIR}/baseline_results_unbalanced.csv")
print("\n==================== NESTED CV RESULTS (OUTER MEANS) ====================")
print(results_df)

for model_name in outer_metrics.keys():
    slug = slug_map[model_name]

    y_true_all = np.concatenate(all_y_true[model_name])
    y_pred_all = np.concatenate(all_y_pred[model_name])
    y_prob_all = np.concatenate(all_y_prob[model_name])

    np.save(f"{OUT_DIR}/{slug}_all_y_true.npy", y_true_all)
    np.save(f"{OUT_DIR}/{slug}_all_y_pred.npy", y_pred_all)
    np.save(f"{OUT_DIR}/{slug}_all_y_prob.npy", y_prob_all)

    cm_global = confusion_matrix(y_true_all, y_pred_all)
    save_confusion_heatmap(
    cm_global,
    f"{OUT_DIR}/{slug}_confusion_matrix_heatmap.png",
    title=f"{model_name} - Global Confusion Matrix"
    )
    np.savetxt(f"{OUT_DIR}/{slug}_confusion_matrix.csv", cm_global, fmt="%d", delimiter=",")

rows = []
for model_name, params_list in best_hyperparams_per_fold.items():
    for params in params_list:
        row = params.copy()
        row["model"] = model_name
        rows.append(row)

hyperparams_df = pd.DataFrame(rows)
hyperparams_df = hyperparams_df.set_index(["model", "fold"]).sort_index()
hyperparams_df.to_csv(os.path.join(OUT_DIR, "baseline_best_hyperparameters_per_fold.csv"))

print("Baseline results saved inside folder:", OUT_DIR)
