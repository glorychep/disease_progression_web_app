
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Helper plotting functions (safe-check single-class)
LEARNING_CURVE_POINTS = 8  # points for learning curve

def plot_confusion_matrix(y_true, y_pred, classes, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path

def plot_roc_curves(y_true, y_proba, classes, title, path):
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] <= 1:
        return None
    plt.figure(figsize=(8,6))
    for i in range(len(classes)):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{classes[i]} (AUC={roc_auc:.2f})")
        except Exception:
            continue
    plt.plot([0,1],[0,1],"k--", lw=2)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(title)
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path

def plot_pr_curves(y_true, y_proba, classes, title, path):
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] <= 1:
        return None
    plt.figure(figsize=(8,6))
    for i in range(len(classes)):
        try:
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap = average_precision_score(y_bin[:, i], y_proba[:, i])
            plt.plot(recall, precision, label=f"{classes[i]} (AP={ap:.2f})")
        except Exception:
              continue
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(loc="best"); plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path

def top_feature_importance(importances, feature_names, k=10, title="Top Features", path="feat.png"):
    idx = np.argsort(importances)[-k:][::-1]
    plt.figure(figsize=(8,5))
    plt.barh(range(len(idx)), importances[idx], align="center")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path

def plot_learning_curve(estimator, X, y, title, path, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1,1.0,LEARNING_CURVE_POINTS)):
    try:
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs, train_sizes=train_sizes)
    except Exception:
        return None
    train_mean = train_scores.mean(axis=1); train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1); test_std = test_scores.std(axis=1)
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, "o-", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel("Training Examples"); plt.ylabel("Accuracy"); plt.title(title); plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    return path

# Data loading
file_path = "Active on ART Patients Linelist_Aug-2025.csv"  # <-- change if your file name differs
FORCE_TARGET = None    # e.g. "Viral_Suppressed" or "Last WHO Stage" — set to None to auto-detect

df=pd.read_csv(file_path)

# Drop PII/common IDs
drop_cols =["Name", "DOB","Date of Birth", "CCC No", "CCC Number",
            "NUPI", "SHA No", "id", "ID","Patient ID"]
df=df.drop(columns=drop_cols,
               errors="ignore")

# Auto-detect target function (viral-load -> Viral_Suppressed else Last WHO Stage)
def detect_and_create_target(df, force_target=None):
    df2 = df.copy()
    if force_target:
        if force_target not in df2.columns:
            raise ValueError(f"FORCE_TARGET '{force_target}' not in columns.")
        return df2, force_target

    vl_candidates = [
        "viral load", "last viral load", "last viral load result", "last viral load copies",
        "vl result", "vl", "last vl", "lastvl", "last_vl", "viral_load", "viral_load_copies"
    ]
    cols_lower = {c.lower(): c for c in df2.columns}

    found_vl_col = None
    for cand in vl_candidates:
        for c_lower, orig in cols_lower.items():
            if cand in c_lower or c_lower in cand:
                found_vl_col = orig
                break
        if found_vl_col:
           break

    if found_vl_col:
        print(f"Detected viral-load column: '{found_vl_col}'. Attempting to create binary 'Viral_Suppressed' (<1000).")
        col = df2[found_vl_col]
        numeric_col = pd.to_numeric(col, errors="coerce")
        if numeric_col.notnull().sum() > 0:
            df2["Viral_Suppressed"] = (numeric_col < 1000).astype(int)
            return df2, "Viral_Suppressed"
        else:
            col_str = col.astype(str).str.lower()
            true_keys = ["suppressed", "undetectable", "not detected", "undetect"]
            false_keys = ["not suppressed", "unsuppressed", "detected"]
            df2["Viral_Suppressed"] = col_str.apply(
                lambda x: 1 if any(k in x for k in true_keys) else (0 if any(k in x for k in false_keys) else None)
            )
            if df2["Viral_Suppressed"].isnull().sum() > len(df2)*0.2:
                print("Mapping ambiguous for many values; falling back to 'Last WHO Stage' if present.")
                if "Last WHO Stage" in df2.columns:
                    return df2, "Last WHO Stage"
                else:
                    fallback = df2.columns[-1]
                    print(f"No WHO stage column. Falling back to '{fallback}'. Please set FORCE_TARGET if incorrect.")
                    return df2, fallback
            df2["Viral_Suppressed"] = df2["Viral_Suppressed"].fillna(df2["Viral_Suppressed"].mode().iloc[0]).astype(int)
            return df2, "Viral_Suppressed"
    else:
        if "Last WHO Stage" in df2.columns:
            print("No viral-load-like column detected. Using 'Last WHO Stage' as target.")
            return df2, "Last WHO Stage"
        else:
            fallback = df2.columns[-1]
            print(f"WARNING: No viral-load/WHO columns detected. Falling back to last column '{fallback}'. Consider setting FORCE_TARGET.")
            return df2, fallback

df, target = detect_and_create_target(df, FORCE_TARGET)

# Drop columns with >70% missing
missing_threshold = 0.70
missing_fraction = df.isnull().mean()
cols_to_drop = missing_fraction[missing_fraction > missing_threshold].index.tolist()
if cols_to_drop:
    print("Dropping high-missing columns:", cols_to_drop)
df = df.drop(columns=cols_to_drop)

# Fill missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

# Encode categorical features (except target for now)
for col in df.select_dtypes(include=["object", "category"]).columns:
    if col == target:
        continue
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Encode target if categorical
if not pd.api.types.is_numeric_dtype(df[target]):
    le_t = LabelEncoder()
    df[target] = le_t.fit_transform(df[target].astype(str))
    target_class_labels = [str(c) for c in le_t.classes_]
else:
    target_class_labels = [str(c) for c in sorted(df[target].unique())]

# Prepare X, y, and scale (for LR)
X = df.drop(columns=[target])
y = df[target].astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (try stratify)
try:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models + RandomizedSearchCV tuning
results = {}
classes_sorted = sorted(np.unique(y))

# A) Random Forest tuning
rf_param_dist = {
    "n_estimators": [100,200,300,400,500],
    "max_depth": [None,10,20,30,40],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4],
    "bootstrap": [True, False]
}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=1, random_state=42)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_proba = rf_best.predict_proba(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cv = cross_val_score(rf_best, X_scaled, y, cv=5, scoring="accuracy")
rf_cv_mean, rf_cv_std = rf_cv.mean(), rf_cv.std()
rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='macro', zero_division=0)
# RF plots
rf_cm = plot_confusion_matrix(y_test, rf_pred, classes_sorted, "Confusion Matrix — Random Forest", "rf_confusion.png")
rf_roc = plot_roc_curves(y_test, rf_proba, classes_sorted, "ROC Curves — Random Forest", "rf_roc.png")
rf_pr  = plot_pr_curves(y_test, rf_proba, classes_sorted, "PR Curves — Random Forest", "rf_pr.png")
rf_feat = top_feature_importance(rf_best.feature_importances_, X.columns, 10, "Top 10 Features — Random Forest", "rf_feat.png")
rf_lc = plot_learning_curve(rf_best, X_scaled, y, "Learning Curve — Random Forest", "rf_learning_curve.png")

results["Random Forest"] = {
    "model": rf_best, "best_params": rf_search.best_params_, "acc": rf_acc,
    "cv_mean": rf_cv_mean, "cv_std": rf_cv_std, "prec_macro": rf_prec, "rec_macro": rf_rec, "f1_macro": rf_f1,
    "cm": rf_cm, "roc": rf_roc, "pr": rf_pr, "feat": rf_feat, "lc": rf_lc,
    "report": classification_report(y_test, rf_pred, target_names=[str(c) for c in classes_sorted], zero_division=0)
}

# B) Logistic Regression (baseline, tuned C)
lr_param_dist = {"C": np.logspace(-3, 2, 10), "penalty": ["l2"], "solver": ["lbfgs"], "max_iter":[1000]}
lr_search = RandomizedSearchCV(LogisticRegression(multi_class="auto"), lr_param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=1, random_state=42)
lr_search.fit(X_train, y_train)
lr_best = lr_search.best_estimator_
lr_pred = lr_best.predict(X_test)
lr_proba = lr_best.predict_proba(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_cv = cross_val_score(lr_best, X_scaled, y, cv=5, scoring="accuracy")
lr_cv_mean, lr_cv_std = lr_cv.mean(), lr_cv.std()
lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average='macro', zero_division=0)
lr_cm = plot_confusion_matrix(y_test, lr_pred, classes_sorted, "Confusion Matrix — Logistic Regression", "lr_confusion.png")
lr_roc = plot_roc_curves(y_test, lr_proba, classes_sorted, "ROC Curves — Logistic Regression", "lr_roc.png")
lr_pr  = plot_pr_curves(y_test, lr_proba, classes_sorted, "PR Curves — Logistic Regression", "lr_pr.png")
lr_lc = plot_learning_curve(lr_best, X_scaled, y, "Learning Curve — Logistic Regression", "lr_learning_curve.png")


results["Logistic Regression"] = {
    "model": lr_best, "best_params": lr_search.best_params_, "acc": lr_acc,
    "cv_mean": lr_cv_mean, "cv_std": lr_cv_std, "prec_macro": lr_prec, "rec_macro": lr_rec, "f1_macro": lr_f1,
    "cm": lr_cm, "roc": lr_roc, "pr": lr_pr, "feat": None, "lc": lr_lc,
    "report": classification_report(y_test, lr_pred, target_names=[str(c) for c in classes_sorted], zero_division=0)
}

# C) XGBoost tuning
xgb_param_dist = {
    "objective": ["binary:logistic" if len(classes_sorted) == 2 else "multi:softmax"],
    "num_class": [len(classes_sorted) if len(classes_sorted) > 2 else None],
    "n_estimators": [100,200,300,400,500],
    "learning_rate": [0.01,0.05,0.1,0.2,0.3],
    "max_depth": [3,5,7,10,12],
    "min_child_weight": [1,3,5,7],
    "gamma": [0,0.1,0.2,0.3,0.4],
    "subsample": [0.6,0.8,1.0],
    "colsample_bytree": [0.6,0.8,1.0],
    "reg_alpha": [0,0.001,0.01,0.1,1,10],
    "reg_lambda": [0,0.001,0.01,0.1,1,10]
}
xgb_search = RandomizedSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss" if len(classes_sorted) == 2 else "merror"),
                                xgb_param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=1, random_state=42)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
xgb_pred = xgb_best.predict(X_test)
xgb_proba = xgb_best.predict_proba(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_cv = cross_val_score(xgb_best, X_scaled, y, cv=5, scoring="accuracy")
xgb_cv_mean, xgb_cv_std = xgb_cv.mean(), xgb_cv.std()
xgb_prec, xgb_rec, xgb_f1, _ = precision_recall_fscore_support(y_test, xgb_pred, average='macro', zero_division=0)

# XGBoost plots
xgb_cm = plot_confusion_matrix(y_test, xgb_pred, classes_sorted, "Confusion Matrix — XGBoost", "xgb_confusion.png")
xgb_roc = plot_roc_curves(y_test, xgb_proba, classes_sorted, "ROC Curves — XGBoost", "xgb_roc.png")
xgb_pr  = plot_pr_curves(y_test, xgb_proba, classes_sorted, "PR Curves — XGBoost", "xgb_pr.png")
xgb_feat = top_feature_importance(xgb_best.feature_importances_, X.columns, 10, "Top 10 Features — XGBoost", "xgb_feat.png")
xgb_lc = plot_learning_curve(xgb_best, X_scaled, y, "Learning Curve — XGBoost", "xgb_learning_curve.png")

results["XGBoost"] = {
    "model": xgb_best, "best_params": xgb_search.best_params_, "acc": xgb_acc,
    "cv_mean": xgb_cv_mean, "cv_std": xgb_cv_std, "prec_macro": xgb_prec, "rec_macro": xgb_rec, "f1_macro": xgb_f1,
    "cm": xgb_cm, "roc": xgb_roc, "pr": xgb_pr, "feat": xgb_feat, "lc": xgb_lc,
    "report": classification_report(y_test, xgb_pred, target_names=[str(c) for c in classes_sorted], zero_division=0)
}

# Save the best model (example: Random Forest) after training
# Choose the best model based on a metric, e.g., macro F1 score
best_model_name = max(results, key=lambda k: results[k]['f1_macro'])
best_model_info = results[best_model_name]
joblib.dump(best_model_info["model"], "best_model.pkl")
st.write(f"Best model saved: {best_model_name}")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

st.title("Disease Progression Prediction")

st.write("Enter patient features to predict disease progression.")

# Load the saved model and scaler
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure the model training and saving steps were completed.")
    st.stop()

# Get feature names from the training data (assuming X is available from the previous steps)
# If X is not available, you would need to save and load feature names separately
feature_names = X.columns.tolist()

input_data = {}
st.sidebar.header("Patient Features")

# Create input widgets for each feature
for feature in feature_names:
    # Determine the appropriate widget type based on the feature's data type in the original df
    # This requires accessing the original df before encoding
    # For simplicity and assuming features in X are numeric after preprocessing, use number_input
    # In a real application, you'd need a more robust way to handle different data types and their original values/categories
    input_data[feature] = st.sidebar.number_input(f"Enter value for {feature}", value=float(df[feature_names].median()[feature])) # Using median from preprocessed data as default


# Create a DataFrame from input data
input_df = pd.DataFrame([input_data])

# Ensure the order of columns matches the training data
try:
    input_df = input_df[feature_names]
except KeyError as e:
    st.error(f"Missing feature in input data: {e}. Please ensure all required features are provided.")
    st.stop()

# Scale the input data
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.stop()

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.subheader("Prediction Results")

        # Assuming binary classification (0 or 1)
        predicted_class = prediction[0]

        # If target was label encoded, you might need to inverse transform to get original labels
        # For this example, assuming the model predicts the encoded labels directly
        # You would need the label encoder used for the target saved and loaded
        # For now, display the predicted encoded class
        st.write(f"Predicted Class (Encoded): {predicted_class}")

        # Display probabilities for each class
        st.write("Prediction Probabilities:")
        # Assuming classes_sorted is available or can be inferred from the loaded model
        # For this example, let's just show probabilities for 0 and 1 if binary
        if hasattr(model, 'classes_') and len(model.classes_) > 1:
            for i, class_prob in enumerate(prediction_proba[0]):
                 st.write(f"Class {model.classes_[i]}: {class_prob:.4f}")
        elif prediction_proba.shape[1] == 1: # Single class case (unlikely for classification but for robustness)
             st.write(f"Class {model.classes_[0] if hasattr(model, 'classes_') else '0'}: {prediction_proba[0][0]:.4f}")
        else: # Handle cases where model.classes_ is not available or prediction_proba structure is unexpected
             st.write("Could not determine class labels for probabilities.")
             for i, class_prob in enumerate(prediction_proba[0]):
                 st.write(f"Probability for class {i}: {class_prob:.4f}")


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

