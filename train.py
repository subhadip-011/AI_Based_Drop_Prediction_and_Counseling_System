import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
df = pd.read_csv("data.csv", sep=";")

# Encode target
label_enc = LabelEncoder()
df["Target_enc"] = label_enc.fit_transform(df["Target"])

X = df.drop(columns=["Target", "Target_enc"], errors="ignore")
y = df["Target_enc"]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Save feature names for app.py alignment
features = X.columns.tolist()

# ---------------------------------------------------------------------
# 2. Train-test split (80:20)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save unseen test data for app demo
X_test.to_csv("demo_data_for_app.csv", index=False)
print("Saved 'demo_data_for_app.csv' for the dashboard demonstration.")

# ---------------------------------------------------------------------
# 3. Scaling
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 4. SMOTE (on training data only)
# ---------------------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("Class distribution after SMOTE:", np.bincount(y_train_resampled))

# ---------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    ),
}

best_model = None
best_acc = 0.0
best_name = ""

# 3-fold Stratified CV
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# scoring dictionary for multi-metric CV
scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}

# ---------------------------------------------------------------------
# 6. CV + Test evaluation for sklearn models (LR, RF, XGB)
# ---------------------------------------------------------------------
for name, model in models.items():
    # Multi-metric cross-validation
    cv_results = cross_validate(
        model,
        X_train_resampled,
        y_train_resampled,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    print(f"\n{name} – 3-fold Cross-Validation:")
    print(
        f"  Accuracy       : {cv_results['test_accuracy'].mean():.4f} "
        f"± {cv_results['test_accuracy'].std():.4f}"
    )
    print(
        f"  Precision_macro: {cv_results['test_precision_macro'].mean():.4f} "
        f"± {cv_results['test_precision_macro'].std():.4f}"
    )
    print(
        f"  Recall_macro   : {cv_results['test_recall_macro'].mean():.4f} "
        f"± {cv_results['test_recall_macro'].std():.4f}"
    )
    print(
        f"  F1_macro       : {cv_results['test_f1_macro'].mean():.4f} "
        f"± {cv_results['test_f1_macro'].std():.4f}"
    )

    # Train on full resampled training data
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on held-out test set
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    print(f"\n{name} – Test Set Performance:")
    print(f"  Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_enc.classes_))

    # Track best model based on test accuracy
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# ---------------------------------------------------------------------
# 7. ANN – 3-fold CV with Accuracy, Precision, Recall, F1
# ---------------------------------------------------------------------
print("\nPerforming Cross-Validation for ANN...")

ann_cv_acc = []
ann_cv_prec = []
ann_cv_rec = []
ann_cv_f1 = []

for train_idx, val_idx in cv.split(X_train_resampled, y_train_resampled):
    X_train_cv, X_val_cv = (
        X_train_resampled[train_idx],
        X_train_resampled[val_idx],
    )
    y_train_cv, y_val_cv = (
        y_train_resampled[train_idx],
        y_train_resampled[val_idx],
    )

    y_train_cv_cat = to_categorical(
        y_train_cv, num_classes=len(label_enc.classes_)
    )

    model = Sequential(
        [
            Input(shape=(X_train_cv.shape[1],)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(len(label_enc.classes_), activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        X_train_cv,
        y_train_cv_cat,
        epochs=15,
        batch_size=32,
        verbose=0,
    )

    # Predictions on validation fold
    y_val_pred_proba = model.predict(X_val_cv, verbose=0)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)

    # Metrics (macro average for multi-class)
    acc_fold = accuracy_score(y_val_cv, y_val_pred)
    prec_fold = precision_score(
        y_val_cv, y_val_pred, average="macro", zero_division=0
    )
    rec_fold = recall_score(
        y_val_cv, y_val_pred, average="macro", zero_division=0
    )
    f1_fold = f1_score(
        y_val_cv, y_val_pred, average="macro", zero_division=0
    )

    ann_cv_acc.append(acc_fold)
    ann_cv_prec.append(prec_fold)
    ann_cv_rec.append(rec_fold)
    ann_cv_f1.append(f1_fold)

print("\nANN – 3-fold Cross-Validation:")
print(
    f"  Accuracy       : {np.mean(ann_cv_acc):.4f} "
    f"± {np.std(ann_cv_acc):.4f}"
)
print(
    f"  Precision_macro: {np.mean(ann_cv_prec):.4f} "
    f"± {np.std(ann_cv_prec):.4f}"
)
print(
    f"  Recall_macro   : {np.mean(ann_cv_rec):.4f} "
    f"± {np.std(ann_cv_rec):.4f}"
)
print(
    f"  F1_macro       : {np.mean(ann_cv_f1):.4f} "
    f"± {np.std(ann_cv_f1):.4f}"
)

# ---------------------------------------------------------------------
# 8. Final ANN training on full resampled train data + test evaluation
# ---------------------------------------------------------------------
y_train_cat = to_categorical(
    y_train_resampled, num_classes=len(label_enc.classes_)
)
y_test_cat = to_categorical(
    y_test, num_classes=len(label_enc.classes_)
)

ann = Sequential(
    [
        Input(shape=(X_train_resampled.shape[1],)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(label_enc.classes_), activation="softmax"),
    ]
)

ann.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

ann.fit(X_train_resampled, y_train_cat, epochs=15, batch_size=32, verbose=0)

loss, acc = ann.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"\nANN (Deep Learning) – Test Accuracy: {acc:.4f}")

# Optionally, get precision/recall/F1 on test set for ANN too
y_test_pred_proba = ann.predict(X_test_scaled, verbose=0)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

ann_test_prec = precision_score(
    y_test, y_test_pred, average="macro", zero_division=0
)
ann_test_rec = recall_score(
    y_test, y_test_pred, average="macro", zero_division=0
)
ann_test_f1 = f1_score(
    y_test, y_test_pred, average="macro", zero_division=0
)

print("\nANN (Deep Learning) – Test Set Detailed Metrics:")
print(f"  Precision_macro: {ann_test_prec:.4f}")
print(f"  Recall_macro   : {ann_test_rec:.4f}")
print(f"  F1_macro       : {ann_test_f1:.4f}")

# update best model if ANN test accuracy is higher
if acc > best_acc:
    best_acc = acc
    best_model = ann
    best_name = "ANN (Deep Learning)"

# ---------------------------------------------------------------------
# 9. Save best model and artifacts
# ---------------------------------------------------------------------
if best_name == "ANN (Deep Learning)":
    best_model.save("dl_model.h5")
    joblib.dump("deep_learning", "model_type.pkl")
else:
    joblib.dump(best_model, "model.pkl")
    joblib.dump("sklearn", "model_type.pkl")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_enc, "label_encoder.pkl")
joblib.dump(features, "features.pkl")

print("\nTraining complete.")
print(f"Best Model: {best_name} with accuracy {best_acc:.4f}")
