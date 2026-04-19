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

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# =========================
# 2. Define target and ID
# =========================
TARGET = "Irrigation_Need"
ID_COL = "id"

X = train.drop(columns=[TARGET]).copy()
y = train[TARGET].copy()
X_test = test.copy()


# =========================
# 3. Identify feature types
# =========================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
feature_cols = [col for col in X.columns if col != ID_COL]
cat_features = [col for col in cat_cols if col != ID_COL]

X = X[feature_cols]
X_test_model = X_test[feature_cols]

print("Categorical columns:", cat_features)
print("Numeric columns:", len([c for c in feature_cols if c not in cat_features]))


# =========================
# 4. Class weights
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
# 5. Threshold tuning helper
# =========================
def apply_class_multipliers(proba: np.ndarray, class_order: np.ndarray, multipliers: dict) -> np.ndarray:
    adjusted = proba.copy()
    for i, cls in enumerate(class_order):
        adjusted[:, i] *= multipliers[cls]
    return class_order[np.argmax(adjusted, axis=1)]


# =========================
# 6. Cross-validation setup
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.empty(len(train), dtype=object)
oof_proba = np.zeros((len(train), len(classes)))
test_preds_proba = []

cv_scores = []


# =========================
# 7. Train across folds
# =========================
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n========== Fold {fold} ==========")

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
        random_seed=42 + fold,
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

    # Validation probabilities
    valid_proba = model.predict_proba(X_valid)
    oof_proba[valid_idx] = valid_proba

    # Validation predictions without threshold tuning
    valid_preds = model.classes_[np.argmax(valid_proba, axis=1)]
    oof_preds[valid_idx] = valid_preds

    fold_score = balanced_accuracy_score(y_valid, valid_preds)
    cv_scores.append(fold_score)

    print(f"Fold {fold} Balanced Accuracy: {fold_score:.6f}")

    # Test probabilities from this fold
    test_fold_proba = model.predict_proba(X_test_model)
    test_preds_proba.append(test_fold_proba)


# =========================
# 8. Baseline CV results
# =========================
overall_score = balanced_accuracy_score(y, oof_preds)
print("\nFold scores:", [round(s, 6) for s in cv_scores])
print("Mean CV Balanced Accuracy:", round(np.mean(cv_scores), 6))
print("OOF Balanced Accuracy:", round(overall_score, 6))


# =========================
# 9. Fine threshold tuning
# =========================
class_order = model.classes_

# Finer search around the area that has already worked for you.
search_space = {
    "Low": np.round(np.arange(0.96, 1.01, 0.01), 2),
    "Medium": np.round(np.arange(0.98, 1.03, 0.01), 2),
    "High": np.round(np.arange(1.58, 1.68, 0.01), 2)
}

best_score = -1
best_multipliers = None
results = []

total_combos = (
    len(search_space["Low"]) *
    len(search_space["Medium"]) *
    len(search_space["High"])
)
print(f"\nSearching {total_combos} threshold combinations...")

counter = 0
for low_mult, med_mult, high_mult in product(
    search_space["Low"],
    search_space["Medium"],
    search_space["High"]
):
    multipliers = {
        "Low": float(low_mult),
        "Medium": float(med_mult),
        "High": float(high_mult)
    }

    tuned_oof_preds = apply_class_multipliers(oof_proba, class_order, multipliers)
    tuned_score = balanced_accuracy_score(y, tuned_oof_preds)

    results.append((tuned_score, multipliers))

    if tuned_score > best_score:
        best_score = tuned_score
        best_multipliers = multipliers

    counter += 1
    if counter % 100 == 0:
        print(f"Checked {counter}/{total_combos} combinations... Current best: {best_score:.6f}")

# Sort and show top 10
results = sorted(results, key=lambda x: x[0], reverse=True)

print("\nTop 10 threshold combinations:")
for rank, (score, multipliers) in enumerate(results[:10], start=1):
    print(f"{rank}. Score={score:.6f} | {multipliers}")

print("\nBest threshold multipliers:", best_multipliers)
print("Best tuned OOF Balanced Accuracy:", round(best_score, 6))


# =========================
# 10. Build final tuned test predictions and save top 3 submissions
# =========================
avg_test_proba = np.mean(test_preds_proba, axis=0)

top_thresholds = [
    {"Low": 1.01, "Medium": 1.02, "High": 1.66},
    {"Low": 1.02, "Medium": 1.02, "High": 1.66},
    {"Low": 0.99, "Medium": 1.00, "High": 1.63},
]

for i, multipliers in enumerate(top_thresholds, start=1):
    test_pred_labels = apply_class_multipliers(
        avg_test_proba,
        class_order,
        multipliers
    )

    submission = sample_submission.copy()
    submission[TARGET] = test_pred_labels

    filename = f"submission_top{i}.csv"
    submission.to_csv(filename, index=False)

    print(f"\nSaved {filename}")
    print(f"Multipliers used: {multipliers}")
    print(submission.head())