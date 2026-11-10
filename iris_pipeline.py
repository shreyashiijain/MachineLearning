"""Iris species classification: Logistic Regression vs Random Forest vs KNN."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

OUT = "/home/claude/"
SEED = 42

# ---------------- Step 1: load, explore, preprocess ----------------
df = pd.read_csv("/mnt/user-data/uploads/Iris.csv")
print("Shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("Duplicate rows (excluding Id):", df.drop(columns="Id").duplicated().sum())
print("\nClass balance:\n", df["Species"].value_counts())

# Id is just a row counter -> carries no signal, drop it
df = df.drop(columns="Id")

# Encode the string target into integers 0/1/2
le = LabelEncoder()
df["Species_enc"] = le.fit_transform(df["Species"])
print("\nEncoding:", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop(columns=["Species", "Species_enc"])
y = df["Species_enc"]

# 80/20 split, stratified so each species keeps the same proportion
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print("\nTrain:", X_train.shape, "Test:", X_test.shape)

# ---------------- Step 2: feature analysis ----------------
corr = df.drop(columns="Species").corr()
print("\nCorrelation with target:\n", corr["Species_enc"].drop("Species_enc").sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix")
plt.tight_layout(); plt.savefig(OUT + "01_correlation.png", dpi=140); plt.close()

rf_fi = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X_train, y_train)
imp = pd.Series(rf_fi.feature_importances_, index=X.columns).sort_values()
print("\nRandom Forest feature importances:\n", imp.sort_values(ascending=False).round(3))

fig, ax = plt.subplots(figsize=(6, 3.5))
imp.plot.barh(ax=ax, color="#4C78A8")
ax.set_title("Random Forest feature importance"); ax.set_xlabel("Importance")
plt.tight_layout(); plt.savefig(OUT + "02_feature_importance.png", dpi=140); plt.close()

pair = sns.pairplot(df.drop(columns="Species_enc"), hue="Species", height=1.8)
pair.savefig(OUT + "03_pairplot.png", dpi=120); plt.close("all")

# ---------------- Step 3: train 3 models ----------------
# LR and KNN are scale/distance sensitive -> StandardScaler inside a pipeline
# (fit on train only, so no leakage). Random Forest does not need scaling.
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}
for m in models.values():
    m.fit(X_train, y_train)

# ---------------- Step 4: evaluate ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, preds = [], {}
for name, m in models.items():
    p = m.predict(X_test); preds[name] = p
    cvs = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, p),
        "Precision (macro)": precision_score(y_test, p, average="macro"),
        "Recall (macro)": recall_score(y_test, p, average="macro"),
        "F1 (macro)": f1_score(y_test, p, average="macro"),
        "5-fold CV acc (mean)": cvs.mean(),
        "CV std": cvs.std(),
    })
res = pd.DataFrame(rows).set_index("Model").round(4)
print(f"\n=== Model comparison (20% hold-out test set, n={len(y_test)}) ===")
print(res.to_string())
res.to_csv(OUT + "model_comparison.csv")

fig, ax = plt.subplots(figsize=(8, 4))
res[["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]].plot.bar(ax=ax, rot=0)
ax.set_ylim(0.8, 1.02); ax.set_title("Model comparison on test set"); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "04_model_comparison.png", dpi=140); plt.close()

# ---------------- Step 5: best model ----------------
# A 30-row test set moves in steps of 3.3% per sample, so test scores tie easily.
# Select on 5-fold CV accuracy (uses all 150 rows), then test F1 as tie-break.
best = res.sort_values(["5-fold CV acc (mean)", "F1 (macro)"], ascending=False).index[0]
print("\nBest model:", best)
print(classification_report(y_test, preds[best], target_names=le.classes_))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, p) in zip(axes, preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, p), display_labels=[c.replace("Iris-", "") for c in le.classes_]
                           ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name + (" (best)" if name == best else ""))
plt.tight_layout(); plt.savefig(OUT + "05_confusion_matrices.png", dpi=140); plt.close()
print("\nConfusion matrix of best model:\n", confusion_matrix(y_test, preds[best]))
