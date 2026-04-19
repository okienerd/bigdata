import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier


# =========================
# 1. Load data
# =========================
train_path = "/Users/jimmyduncan/Desktop/Kaggle/playground-series-s6e4/train.csv"
test_path = "/Users/jimmyduncan/Desktop/Kaggle/playground-series-s6e4/test.csv"
sample_path = "/Users/jimmyduncan/Desktop/Kaggle/playground-series-s6e4/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

TARGET = "Irrigation_Need"
ID_COL = "id"

X = train.drop(columns=[TARGET]).copy()
y = train[TARGET].copy()
X_test = test.copy()


# =========================
# 2. Feature setup
# =========================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
feature_cols = [col for col in X.columns if col != ID_COL]
cat_features = [col for col in cat_cols if col != ID_COL]

X = X[feature_cols]
X_test_model = X_test[feature_cols]

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Categorical columns:", cat_features)


# =========================
# 3. Class weights
# =========================
classes = np.unique(y)
auto_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y
)
class_weights = dict(zip(classes, auto_weights))

print("Class weights:", class_weights)
print("Target distribution:")
print(y.value_counts(normalize=True))


# =========================
# 4. Helper function
# =========================
def apply_class_multipliers(proba: np.ndarray, class_order: np.ndarray, multipliers: dict) -> np.ndarray:
    adjusted = proba.copy()
    for i, cls in enumerate(class_order):
        adjusted[:, i] *= multipliers[cls]
    return class_order[np.argmax(adjusted, axis=1)]


# =========================
# 5. Seed averaging setup
# =========================
seeds = [42, 99, 2024]
n_splits = 5

# Keep folds fixed so we only change the model seed
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_proba_total = np.zeros((len(train), len(classes)))
test_proba_total = np.zeros((len(test), len(classes)))

seed_scores = []


# =========================
# 6. Train with multiple seeds
# =========================
for seed in seeds:
    print(f"\n================ SEED {seed} ================")

    oof_proba_seed = np.zeros((len(train), len(classes)))
    test_proba_seed = np.zeros((len(test), len(classes)))
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n---------- Fold {fold} ----------")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="Accuracy",
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5,
            random_strength=1,
            bagging_temperature=1,
            border_count=254,
            class_weights=class_weights,
            verbose=200,
            random_seed=seed + fold,
            od_type="Iter",
            od_wait=100,
            thread_count=-1
        )

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_valid, y_valid),
            use_best_model=True
        )

        valid_proba = model.predict_proba(X_valid)
        oof_proba_seed[valid_idx] = valid_proba

        valid_pred = model.classes_[np.argmax(valid_proba, axis=1)]
        fold_score = balanced_accuracy_score(y_valid, valid_pred)
        fold_scores.append(fold_score)

        print(f"Fold {fold} Balanced Accuracy: {fold_score:.6f}")

        test_fold_proba = model.predict_proba(X_test_model)
        test_proba_seed += test_fold_proba / n_splits

    seed_oof_pred = model.classes_[np.argmax(oof_proba_seed, axis=1)]
    seed_score = balanced_accuracy_score(y, seed_oof_pred)
    seed_scores.append(seed_score)

    print(f"\nSeed {seed} fold scores: {[round(s, 6) for s in fold_scores]}")
    print(f"Seed {seed} OOF Balanced Accuracy: {seed_score:.6f}")

    oof_proba_total += oof_proba_seed / len(seeds)
    test_proba_total += test_proba_seed / len(seeds)


# =========================
# 7. Baseline seed-averaged score
# =========================
class_order = model.classes_
baseline_oof_pred = class_order[np.argmax(oof_proba_total, axis=1)]
baseline_score = balanced_accuracy_score(y, baseline_oof_pred)

print("\nSeed OOF scores:", [round(s, 6) for s in seed_scores])
print("Seed-averaged baseline OOF Balanced Accuracy:", round(baseline_score, 6))


# =========================
# 8. Threshold tuning
# =========================
search_space = {
    "Low": [0.95, 1.00, 1.05],
    "Medium": [0.95, 1.00, 1.05, 1.10],
    "High": [1.00, 1.20, 1.40, 1.60]
}

best_score = -1
best_multipliers = None

for low_mult, med_mult, high_mult in product(
    search_space["Low"],
    search_space["Medium"],
    search_space["High"]
):
    multipliers = {
        "Low": low_mult,
        "Medium": med_mult,
        "High": high_mult
    }

    tuned_oof_preds = apply_class_multipliers(oof_proba_total, class_order, multipliers)
    tuned_score = balanced_accuracy_score(y, tuned_oof_preds)

    if tuned_score > best_score:
        best_score = tuned_score
        best_multipliers = multipliers

print("\nBest threshold multipliers:", best_multipliers)
print("Best tuned OOF Balanced Accuracy:", round(best_score, 6))


# =========================
# 9. BLEND MODELS FIRST
# =========================

# model A = your BEST model predictions
# model B = your seed averaging predictions

final_test_preds = 0.6 * test_preds_model_A + 0.4 * test_preds_model_B

# =========================
# 10. Apply threshold AFTER blending
# =========================
test_pred_labels = apply_class_multipliers(
    final_test_preds,
    class_order,
    best_multipliers
)


# =========================
# 11. Save submission
# =========================
submission = sample_submission.copy()
submission[TARGET] = test_pred_labels

submission.to_csv("submission_seed_avg_fixed_blending.csv", index=False)

print("\nSaved submission_seed_avg_fixed_blending.csv")
print(submission.head())