"""Profile each candidate dataset for the ALU summative assignment."""
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

def profile(name, df, target):
    print("=" * 80)
    print(f"DATASET: {name}")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Categorical ({len(cat)}): {cat}")
    print(f"Numerical   ({len(num)}): {num}")
    print(f"\nDtypes:\n{df.dtypes.value_counts().to_dict()}")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print(f"\nMissing values: {'NONE' if miss.empty else miss.to_dict()}")
    print(f"Duplicated rows: {df.duplicated().sum()}")
    if target in df.columns:
        vc = df[target].value_counts()
        print(f"\nTarget '{target}' distribution:")
        print((pd.DataFrame({'count': vc, 'pct': (vc / len(df) * 100).round(1)})).to_string())
    print()

# 1. xAPI
xapi = pd.read_csv("xapi/xAPI-Edu-Data.csv")
profile("xAPI Educational Mining (Kaggle)", xapi, "Class")
print("Topic (subject) distribution:", xapi["Topic"].value_counts().to_dict())
print("Behavioral feature stats:")
print(xapi[["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]].describe().round(1).to_string())
print()

# 2. UCI Student Performance
mat = pd.read_csv("uci_student/student-mat.csv", sep=";")
por = pd.read_csv("uci_student/student-por.csv", sep=";")
profile("UCI Student Performance - Math", mat, "G3")
profile("UCI Student Performance - Portuguese", por, "G3")
for nm, d in [("mat", mat), ("por", por)]:
    g3 = d["G3"]
    bins = pd.cut(g3, bins=[-1, 9, 13, 20], labels=["Low(0-9)", "Mid(10-13)", "High(14-20)"])
    print(f"{nm} G3 binned 3-class:", bins.value_counts().to_dict(), "| zeros in G3:", (g3 == 0).sum())
# overlap students (merge keys per dataset docs)
keys = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]
merged = mat.merge(por, on=keys, suffixes=("_mat", "_por"))
print("Students present in BOTH math and portuguese:", len(merged))
print()

# 3a. StudentsPerformance (exams)
sp = pd.read_csv("kaggle_perf/StudentsPerformance.csv")
profile("Kaggle: Students Performance in Exams", sp, "math score")
# 3b. Student Performance Factors
spf = pd.read_csv("kaggle_perf/StudentPerformanceFactors.csv")
profile("Kaggle: Student Performance Factors", spf, "Exam_Score")
print("Exam_Score describe:", spf["Exam_Score"].describe().round(1).to_dict())

# 4. OULAD
info = pd.read_csv("oulad/studentInfo.csv")
profile("OULAD studentInfo", info, "final_result")
print("Other OULAD tables:")
for t in ["assessments", "courses", "studentAssessment", "studentRegistration", "vle"]:
    d = pd.read_csv(f"oulad/{t}.csv")
    print(f"  {t}: {d.shape}, cols={list(d.columns)}")
# studentVle is huge - just count rows cheaply
import subprocess
n = subprocess.run(["wc", "-l", "oulad/studentVle.csv"], capture_output=True, text=True).stdout.split()[0]
print(f"  studentVle: {n} rows (clickstream)")
print("\nOULAD unique students:", info["id_student"].nunique(), "| rows (student-course):", len(info))
print("Modules:", info["code_module"].value_counts().to_dict())
