"""OULAD feasibility baseline: early identification (first 4 weeks) of final outcome."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
EARLY_DAYS = 28  # first 4 weeks of the course

# ---- 1. Aggregate clickstream (chunked; 10.6M rows) ----
print("Aggregating studentVle (first 4 weeks)...")
aggs = []
dtypes = {"code_module": "category", "code_presentation": "category",
          "id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int16"}
for chunk in pd.read_csv("oulad/studentVle.csv", dtype=dtypes, chunksize=2_000_000):
    early = chunk[chunk["date"] < EARLY_DAYS]
    g = early.groupby(["code_module", "code_presentation", "id_student"], observed=True).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        distinct_materials=("id_site", "nunique"),
        first_access=("date", "min"),
    ).reset_index()
    aggs.append(g)
vle = pd.concat(aggs).groupby(["code_module", "code_presentation", "id_student"], observed=True).agg(
    total_clicks=("total_clicks", "sum"),
    active_days=("active_days", "sum"),       # approximate across chunk boundaries
    distinct_materials=("distinct_materials", "sum"),
    first_access=("first_access", "min"),
).reset_index()
print(f"VLE aggregate: {vle.shape}")

# ---- 2. Early assessment scores (submitted within first 4 weeks) ----
sa = pd.read_csv("oulad/studentAssessment.csv", na_values="?")
ass = pd.read_csv("oulad/assessments.csv", na_values="?")
sa = sa.merge(ass[["id_assessment", "code_module", "code_presentation", "date"]], on="id_assessment")
early_sa = sa[sa["date_submitted"] < EARLY_DAYS]
ea = early_sa.groupby(["code_module", "code_presentation", "id_student"]).agg(
    early_score_mean=("score", "mean"), early_n_assess=("score", "size")).reset_index()
print(f"Early assessments: {ea.shape}")

# ---- 3. Build modeling table ----
info = pd.read_csv("oulad/studentInfo.csv", na_values="?")
reg = pd.read_csv("oulad/studentRegistration.csv", na_values="?")
df = info.merge(reg, on=["code_module", "code_presentation", "id_student"], how="left")
df = df.merge(vle, on=["code_module", "code_presentation", "id_student"], how="left")
df = df.merge(ea, on=["code_module", "code_presentation", "id_student"], how="left")

# Exclude students who withdrew before the 4-week observation window ended
df = df[~(df["date_unregistration"] < EARLY_DAYS)].copy()
print(f"Modeling table after exclusions: {df.shape}")

# Feature engineering (baseline-level)
df["registered_early_by"] = -df["date_registration"].fillna(0)  # days before course start
for c in ["total_clicks", "active_days", "distinct_materials"]:
    df[c] = df[c].fillna(0)
df["no_vle_activity"] = (df["total_clicks"] == 0).astype(int)
df["first_access"] = df["first_access"].fillna(EARLY_DAYS)  # never accessed -> max
df["early_score_mean"] = df["early_score_mean"].fillna(df["early_score_mean"].median())
df["early_n_assess"] = df["early_n_assess"].fillna(0)
df["imd_band"] = df["imd_band"].str.replace("10-20", "10-20%", regex=False)
imd_order = {f"{a}-{b}%": i for i, (a, b) in enumerate(
    [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,100)])}
df["imd_band_ord"] = df["imd_band"].map(imd_order)
df["imd_missing"] = df["imd_band_ord"].isna().astype(int)
df["imd_band_ord"] = df["imd_band_ord"].fillna(df["imd_band_ord"].median())
edu_order = {"No Formal quals":0, "Lower Than A Level":1, "A Level or Equivalent":2,
             "HE Qualification":3, "Post Graduate Qualification":4}
df["education_ord"] = df["highest_education"].map(edu_order)
age_order = {"0-35":0, "35-55":1, "55<=":2}
df["age_ord"] = df["age_band"].map(age_order)
df["is_stem"] = df["code_module"].isin(["CCC","DDD","EEE","FFF"]).astype(int)

cat_oh = pd.get_dummies(df[["gender", "region", "disability", "code_module"]], drop_first=True)
num_cols = ["num_of_prev_attempts", "studied_credits", "registered_early_by",
            "total_clicks", "active_days", "distinct_materials", "first_access",
            "no_vle_activity", "early_score_mean", "early_n_assess",
            "imd_band_ord", "imd_missing", "education_ord", "age_ord", "is_stem"]
X = pd.concat([df[num_cols].reset_index(drop=True), cat_oh.reset_index(drop=True)], axis=1)
y = df["final_result"].reset_index(drop=True)
print(f"X: {X.shape}, classes: {y.value_counts().to_dict()}")

# ---- 4. Split 70/15/15 stratified ----
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"train {X_tr.shape[0]} / val {X_val.shape[0]} / test {X_te.shape[0]}")

# ---- 5. Baselines ----
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced")
rf.fit(X_tr, y_tr)
pred = rf.predict(X_te)
proba = rf.predict_proba(X_te)
print("\n=== Random Forest (test) ===")
print(f"Accuracy: {accuracy_score(y_te, pred):.4f}  Macro-F1: {f1_score(y_te, pred, average='macro'):.4f}")
print(f"ROC-AUC (OvR macro): {roc_auc_score(y_te, proba, multi_class='ovr', average='macro'):.4f}")
print(classification_report(y_te, pred, digits=3))

sc = StandardScaler().fit(X_tr)
lr = LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")
lr.fit(sc.transform(X_tr), y_tr)
pred_lr = lr.predict(sc.transform(X_te))
proba_lr = lr.predict_proba(sc.transform(X_te))
print("=== Logistic Regression (test) ===")
print(f"Accuracy: {accuracy_score(y_te, pred_lr):.4f}  Macro-F1: {f1_score(y_te, pred_lr, average='macro'):.4f}")
print(f"ROC-AUC (OvR macro): {roc_auc_score(y_te, proba_lr, multi_class='ovr', average='macro'):.4f}")
print(classification_report(y_te, pred_lr, digits=3))

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 12 RF feature importances:")
print(imp.head(12).round(4).to_string())

# ---- 6. Plots ----
fig, ax = plt.subplots(figsize=(7, 4))
y.value_counts().reindex(["Distinction","Pass","Fail","Withdrawn"]).plot.bar(ax=ax, color="#4C72B0")
ax.set_title("OULAD final_result distribution (after early-withdrawal exclusion)")
ax.set_ylabel("students"); plt.tight_layout(); plt.savefig("oulad_class_distribution.png", dpi=110)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation heatmap — engineered numerical features")
plt.tight_layout(); plt.savefig("oulad_correlation_heatmap.png", dpi=110)
print("\nSaved oulad_class_distribution.png and oulad_correlation_heatmap.png")
