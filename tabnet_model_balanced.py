import numpy as np
import pandas as pd
import optuna
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_OUTER = 3
N_INNER = 3

OUT_DIR = "tabnet_results_balanced"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
df = pd.read_csv("SCF2022_fragility.csv")

# X: predictors, y: target
X = df.drop(columns=["fragile", "YY1"])
y = df["fragile"].astype(int)

feature_names = X.columns.tolist()

outer_cv = StratifiedKFold(
    n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE
)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC AUC can fail if only one class is present in y_true
    try:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["ROC_AUC"] = np.nan
    return metrics

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

def objective_tabnet(trial, X_train, y_train, cv_inner, base_pos_weight):

    # -------- Hyperparameter search space --------
    # Tune how strong we weight the positive class
    pos_weight_trial = trial.suggest_float(
        "pos_weight",
        1.0 * base_pos_weight,
        2.0 * base_pos_weight
    )

    batch_size = trial.suggest_categorical("batch_size", [1024, 2048])
    virtual_batch_size = max(batch_size // 8, 64)

    params = dict(
        mask_type=trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
        n_d=trial.suggest_int("n_d", 8, 52, step=2),
        n_a=trial.suggest_int("n_a", 8, 52, step=2),
        n_steps=trial.suggest_int("n_steps", 2, 8),
        gamma=trial.suggest_float("gamma", 1.0, 2.0),
        lambda_sparse=trial.suggest_float(
            "lambda_sparse", 1e-6, 1e-4, log=True
        ),
        optimizer_params=dict(
            lr=trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        ),
        optimizer_fn=torch.optim.Adam,
        seed=RANDOM_STATE,
        verbose=0,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    inner_scores = []

    for tr_idx, val_idx in cv_inner.split(X_train, y_train):

        scaler_inner = StandardScaler()
        X_tr_raw = X_train.iloc[tr_idx].values.astype(np.float32)
        X_val_raw = X_train.iloc[val_idx].values.astype(np.float32)

        X_tr = scaler_inner.fit_transform(X_tr_raw).astype(np.float32)
        X_val = scaler_inner.transform(X_val_raw).astype(np.float32)

        y_tr = y_train.iloc[tr_idx].values.astype(np.int64)
        y_val = y_train.iloc[val_idx].values.astype(np.int64)

        # sample weights
        w_tr = np.where(y_tr == 1, pos_weight_trial, 1.0)

        model = TabNetClassifier(**params)

        model.fit(
            X_train=X_tr,
            y_train=y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            weights=w_tr,
            max_epochs=300,
            patience=60,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )

        y_val_prob = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_val_prob)
        inner_scores.append(score)

    return np.mean(inner_scores)


# -------------------------------------------------------
# Nested Cross-Validation
# -------------------------------------------------------
outer_metrics = {"TabNet": []}

all_y_true = []
all_y_pred = []
all_y_prob = []

feature_importances_folds = []
best_hyperparams_per_fold = []

fold_idx = 0
for train_idx, test_idx in outer_cv.split(X, y):
    fold_idx += 1
    print(f"\n================ OUTER FOLD {fold_idx} / {N_OUTER} ================")

    X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # base imbalance handling
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    base_pos_weight = neg / pos

    cv_inner = StratifiedKFold(
        n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE
    )

    # ---------- Optuna inner CV ----------
    print(f"[Fold {fold_idx}] Optuna: TabNet hyperparameter tuning")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_tabnet(
            trial, X_tr, y_tr, cv_inner, base_pos_weight
        ),
        n_trials=30,
        show_progress_bar=False,
    )

    best_params = study.best_params
    print("Best params:", best_params)
    print("Best inner ROC_AUC:", study.best_value)

    best_params_record = best_params.copy()
    best_params_record["best_inner_ROC_AUC"] = study.best_value
    best_params_record["fold"] = fold_idx
    best_hyperparams_per_fold.append(best_params_record)

    # ---------- Retrain final model on full outer train ----------
    final_model = TabNetClassifier(
        n_d=best_params["n_d"],
        n_a=best_params["n_a"],
        n_steps=best_params["n_steps"],
        gamma=best_params["gamma"],
        lambda_sparse=best_params["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": best_params["lr"]},
        mask_type=best_params["mask_type"],
        seed=RANDOM_STATE,
        verbose=10,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    batch_size_final = best_params["batch_size"]
    virtual_batch_size_final = max(batch_size_final // 8, 64)
    pos_weight_final = best_params["pos_weight"]

    scaler_outer = StandardScaler()
    X_tr_np_raw = X_tr.values.astype(np.float32)
    X_te_np_raw = X_te.values.astype(np.float32)

    X_tr_np = scaler_outer.fit_transform(X_tr_np_raw).astype(np.float32)
    X_te_np = scaler_outer.transform(X_te_np_raw).astype(np.float32)

    y_tr_np = y_tr.values.astype(np.int64)
    y_te_np = y_te.values.astype(np.int64)
    w_tr_full = np.where(y_tr_np == 1, pos_weight_final, 1.0)

    final_model.fit(
        X_train=X_tr_np,
        y_train=y_tr_np,
        eval_set=[(X_tr_np, y_tr_np)],
        eval_metric=["auc"],
        weights=w_tr_full,
        max_epochs=300,
        patience=60,
        batch_size=batch_size_final,
        virtual_batch_size=virtual_batch_size_final,
        num_workers=0,
        drop_last=False,
    )

    fold_importances = final_model.feature_importances_

    top10_idx = np.argsort(fold_importances)[::-1][:10]
    top10_features = [feature_names[i] for i in top10_idx]
    top10_values = fold_importances[top10_idx]

    feature_importances_folds.append({
        "fold": fold_idx,
        "features": top10_features,
        "importances": top10_values
    })

    y_prob = final_model.predict_proba(X_te_np)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics_fold = compute_metrics(y_te_np, y_pred, y_prob)
    outer_metrics["TabNet"].append(metrics_fold)

    all_y_true.append(y_te_np)
    all_y_pred.append(y_pred)
    all_y_prob.append(y_prob)

# -------------------------------------------------------
# Results
# -------------------------------------------------------
summary = {
    "TabNet": {
        m: np.mean([fold[m] for fold in outer_metrics["TabNet"]])
        for m in outer_metrics["TabNet"][0].keys()
    }
}

results_df = pd.DataFrame(summary).T
print("\n==================== NESTED CV RESULTS (OUTER MEANS) ====================")
print(results_df)

# Global confusion matrix
all_y_true_arr = np.concatenate(all_y_true)
all_y_pred_arr = np.concatenate(all_y_pred)
all_y_prob_arr = np.concatenate(all_y_prob)

cm_global = confusion_matrix(all_y_true_arr, all_y_pred_arr)

save_confusion_heatmap(
    cm_global,
    os.path.join(OUT_DIR, "tabnet_confusion_matrix_heatmap.png"),
    title="TabNet - Global Confusion Matrix"
)

np.savetxt(
    os.path.join(OUT_DIR, "tabnet_confusion_matrix.csv"),
    cm_global,
    delimiter=",",
    fmt="%d"
)

# Mean metrics
results_df.to_csv(os.path.join(OUT_DIR, "tabnet_results_balanced.csv"), index=True)

# Global predictions & labels
np.save(os.path.join(OUT_DIR, "tabnet_all_y_true.npy"), all_y_true_arr)
np.save(os.path.join(OUT_DIR, "tabnet_all_y_pred.npy"), all_y_pred_arr)
np.save(os.path.join(OUT_DIR, "tabnet_all_y_prob.npy"), all_y_prob_arr)

# Save feature importances
rows = []
mean_dict = {}

for entry in feature_importances_folds:
    fold = entry["fold"]
    feats = entry["features"]
    vals = entry["importances"]

    for f, v in zip(feats, vals):
        rows.append({
            "fold": fold,
            "feature": f,
            "importance": v
        })

        if f not in mean_dict:
            mean_dict[f] = []
        mean_dict[f].append(v)

mean_importances = {f: np.mean(vs) for f, vs in mean_dict.items()}

sorted_feats = sorted(mean_importances.items(), key=lambda x: x[1], reverse=True)
top10_global = sorted_feats[:10]

for f, imp in top10_global:
    rows.append({
        "fold": "mean",
        "feature": f,
        "importance": imp
    })

fi_combined_df = pd.DataFrame(rows)
fi_combined_df.to_csv(os.path.join(OUT_DIR, "tabnet_top10_feature_importances_combined.csv"),
                      index=False)

# Save best hyperparameters per fold
hyperparams_df = pd.DataFrame(best_hyperparams_per_fold)
hyperparams_df = hyperparams_df.set_index("fold").sort_index()
hyperparams_df.to_csv(os.path.join(OUT_DIR, "tabnet_best_hyperparameters_per_fold.csv"))

print(f"\nSaved TabNet results, feature importances, and hyperparameters to folder: {OUT_DIR}")
