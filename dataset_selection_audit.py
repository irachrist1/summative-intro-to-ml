from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
PROCESSED.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

XAPI_PATH = Path.home() / ".cache/kagglehub/datasets/aljarah/xAPI-Edu-Data/versions/6/xAPI-Edu-Data.csv"
KAGGLE_PERF_PATH = Path.home() / ".cache/kagglehub/datasets/rabieelkharoua/students-performance-dataset/versions/2/Student_performance_data _.csv"


def summarize_df(name, df, target):
    features = [c for c in df.columns if c != target]
    cat_cols = [c for c in features if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in features if c not in cat_cols]
    return {
        "name": name,
        "shape": list(df.shape),
        "target": target,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "target_distribution": df[target].value_counts(dropna=False).to_dict(),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if int(v) > 0},
        "categorical_feature_count": len(cat_cols),
        "numerical_feature_count": len(num_cols),
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
    }


def bin_g3(value):
    if value <= 9:
        return "Low"
    if value <= 13:
        return "Medium"
    return "High"


def load_xapi():
    return pd.read_csv(XAPI_PATH)


def load_uci_combined():
    mat = pd.read_csv(RAW / "uci_student/student-mat.csv", sep=";")
    por = pd.read_csv(RAW / "uci_student/student-por.csv", sep=";")
    mat["subject"] = "Math"
    por["subject"] = "Portuguese"
    df = pd.concat([mat, por], ignore_index=True)
    df["performance_tier"] = df["G3"].map(bin_g3)
    return df


def load_kaggle_perf():
    df = pd.read_csv(KAGGLE_PERF_PATH)
    df["GradeClass"] = df["GradeClass"].astype(int).astype(str)
    return df


def build_oulad_early(days=30):
    out = PROCESSED / f"oulad_early{days}.csv"
    if out.exists():
        return pd.read_csv(out)

    info = pd.read_csv(RAW / "oulad/studentInfo.csv")
    reg = pd.read_csv(RAW / "oulad/studentRegistration.csv")
    keys = ["code_module", "code_presentation", "id_student"]
    df = info.merge(reg[keys + ["date_registration"]], on=keys, how="left")

    chunks = []
    usecols = keys + ["id_site", "date", "sum_click"]
    for chunk in pd.read_csv(RAW / "oulad/studentVle.csv", usecols=usecols, chunksize=1_000_000):
        early = chunk[(chunk["date"] >= 0) & (chunk["date"] <= days)]
        if early.empty:
            continue
        agg = early.groupby(keys).agg(
            total_clicks_30d=("sum_click", "sum"),
            active_days_30d=("date", "nunique"),
            distinct_sites_30d=("id_site", "nunique"),
            interaction_rows_30d=("sum_click", "size"),
        )
        chunks.append(agg)

    if chunks:
        vle = pd.concat(chunks).groupby(level=list(range(len(keys)))).sum().reset_index()
    else:
        vle = pd.DataFrame(columns=keys + ["total_clicks_30d", "active_days_30d", "distinct_sites_30d", "interaction_rows_30d"])

    df = df.merge(vle, on=keys, how="left")
    for col in ["total_clicks_30d", "active_days_30d", "distinct_sites_30d", "interaction_rows_30d"]:
        df[col] = df[col].fillna(0)
    df["avg_clicks_per_active_day_30d"] = np.where(
        df["active_days_30d"] > 0,
        df["total_clicks_30d"] / df["active_days_30d"],
        0,
    )
    df.to_csv(out, index=False)
    return df


def make_preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )


def baseline(df, target, drop_cols=None, model_name="random_forest"):
    drop_cols = drop_cols or []
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns])
    y = df[target].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1)
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    pipe = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", clf)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "report": classification_report(y_test, pred, zero_division=0),
        "test_size": int(len(y_test)),
    }


def save_chosen_plots(df):
    order = df["final_result"].value_counts().index
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="final_result", order=order)
    plt.title("OULAD final_result class distribution")
    plt.xlabel("Final result")
    plt.ylabel("Student-module records")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "oulad_class_distribution.png", dpi=160)
    plt.close()

    numeric = df.select_dtypes(include=[np.number]).drop(columns=["id_student"], errors="ignore")
    corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=False)
    plt.title("OULAD early-window numeric feature correlations")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "oulad_numeric_correlation_heatmap.png", dpi=160)
    plt.close()


def main():
    xapi = load_xapi()
    uci = load_uci_combined()
    kaggle_perf = load_kaggle_perf()
    oulad = build_oulad_early()

    summaries = {
        "xapi": summarize_df("xAPI Educational Mining Dataset", xapi, "Class"),
        "uci_student_combined": summarize_df("UCI Student Performance combined Math/Portuguese", uci, "performance_tier"),
        "kaggle_students_performance": summarize_df("Kaggle Students Performance Dataset", kaggle_perf, "GradeClass"),
        "oulad_early30": summarize_df("OULAD early 30-day engineered table", oulad, "final_result"),
    }

    baselines = {
        "xapi_rf": baseline(xapi, "Class"),
        "uci_rf_no_final_grade_leakage": baseline(uci.drop(columns=["G3"]), "performance_tier", drop_cols=["G1", "G2"]),
        "kaggle_rf_no_gpa_or_id": baseline(kaggle_perf, "GradeClass", drop_cols=["StudentID", "GPA"]),
        "oulad_rf_early30": baseline(oulad, "final_result", drop_cols=["id_student"]),
    }

    save_chosen_plots(oulad)

    output = {"summaries": summaries, "baselines": baselines}
    (ARTIFACTS / "dataset_audit_summary.json").write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
