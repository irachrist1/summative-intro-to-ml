# Dataset Recommendation for ALU Intro to ML Summative

## Recommendation

Use the **Open University Learning Analytics Dataset (OULAD)**, engineered as an early-course multi-class classification task:

> Predict `final_result` (`Distinction`, `Pass`, `Fail`, `Withdrawn`) from demographics, deprivation band, module/presentation context, registration timing, and early VLE engagement features from days 0-30.

This is the best fit for the rubric because it is large enough for both Scikit-learn and TensorFlow, has meaningful behavioral engagement data, supports multi-class confusion matrices and ROC curves, and gives a strong education-equity story through `imd_band`, region, disability, prior attempts, and early online engagement.

The main caution is that `Distinction` is the smallest class, so the project should explicitly address class imbalance and high-aptitude recall.

## Sources Downloaded

- xAPI Educational Mining Dataset: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- UCI Student Performance Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
- Kaggle Students Performance Dataset: https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
- OULAD: https://analyse.kmi.open.ac.uk/open-dataset

## Candidate Audit

### 1. xAPI Educational Mining Dataset

- Shape: 480 rows x 17 columns
- Target: `Class`
- Target distribution: `M` 211, `H` 142, `L` 127
- Missing values: none
- Feature types: 12 categorical, 4 numerical
- Numerical features: `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`
- Categorical features: `gender`, `NationalITy`, `PlaceofBirth`, `StageID`, `GradeID`, `SectionID`, `Topic`, `Semester`, `Relation`, `ParentAnsweringSurvey`, `ParentschoolSatisfaction`, `StudentAbsenceDays`
- Data quality: clean and highly interpretable, but very small
- Classification support: strong 3-class target
- ML/DL feasibility: good for traditional ML; weak for robust deep learning because only 480 records
- Experiment potential: good preprocessing and feature-engineering story, limited hyperparameter/generalization evidence
- Baseline RF: accuracy 0.771, weighted F1 0.770

### 2. UCI Student Performance Dataset

- Shape used: 1,044 rows x 35 columns after stacking Math and Portuguese with a `subject` feature
- Target: `performance_tier`, binned from `G3` as Low/Medium/High
- Target distribution: `Medium` 520, `High` 294, `Low` 230
- Missing values: none
- Feature types: 18 categorical, 16 numerical
- Key features: demographics, parent education/jobs, study time, failures, support, activities, internet access, absences, `G1`, `G2`, `G3`, `subject`
- Data quality: clean, but `G1`/`G2` are very strong grade-history features and can become leakage depending on project framing
- Classification support: good after binning
- ML/DL feasibility: acceptable but still small for deep learning
- Experiment potential: good for binning, leakage-control, subject-aware experiments, feature ablation
- Baseline RF without `G1`, `G2`, `G3`: accuracy 0.545, weighted F1 0.521

### 3. Kaggle Students Performance Dataset

- Shape: 2,392 rows x 15 columns
- Target: `GradeClass`
- Target distribution: `4` 1211, `3` 414, `2` 391, `1` 269, `0` 107
- Missing values: none
- Feature types: all numeric/coded features
- Features: `StudentID`, `Age`, `Gender`, `Ethnicity`, `ParentalEducation`, `StudyTimeWeekly`, `Absences`, `Tutoring`, `ParentalSupport`, `Extracurricular`, `Sports`, `Music`, `Volunteering`, `GPA`, `GradeClass`
- Data quality: clean, but `StudentID` is non-predictive and `GPA` leaks the target relationship
- Classification support: strong 5-class target, though imbalanced
- ML/DL feasibility: better size than xAPI/UCI; still not especially rich
- Experiment potential: good for imbalance and leakage discussion, weaker for subject-specific strengths
- Baseline RF without `StudentID` and `GPA`: accuracy 0.704, weighted F1 0.698

### 4. OULAD

- Engineered shape: 32,593 rows x 18 columns using student info, registration, and VLE interactions from days 0-30
- Target: `final_result`
- Target distribution: `Pass` 12,361, `Withdrawn` 10,156, `Fail` 7,052, `Distinction` 3,024
- Missing values: `imd_band` 1,111; `date_registration` 45
- Feature types: 8 categorical, 9 numerical
- Categorical features: `code_module`, `code_presentation`, `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`
- Numerical features: `id_student`, `num_of_prev_attempts`, `studied_credits`, `date_registration`, `total_clicks_30d`, `active_days_30d`, `distinct_sites_30d`, `interaction_rows_30d`, `avg_clicks_per_active_day_30d`
- Data quality: rich and realistic; needs aggregation and careful leakage control
- Classification support: strong 4-class target with an interpretable high-aptitude class (`Distinction`)
- ML/DL feasibility: best candidate for Scikit-learn and TensorFlow comparison
- Experiment potential: excellent; can vary early time windows, click aggregations, assessment features, imbalance handling, model families, and module-specific analyses
- Baseline RF, early 30-day features only: accuracy 0.538, weighted F1 0.494

## Why OULAD Wins

OULAD best satisfies the technical rubric and the problem story. It supports a credible education-equity framing because students have socioeconomic deprivation bands (`imd_band`), regional context, disability status, prior attempts, and observable behavioral engagement through VLE interactions. The `code_module` feature gives a subject/module-specific angle, while `Distinction` can be framed as high aptitude or exceptional performance.

xAPI is conceptually beautiful for teacher-observable behavior, but 480 rows is too thin for the required TensorFlow Sequential and Functional API comparison. UCI is clean and subject-aware, but less behavioral. The larger Kaggle performance dataset is convenient, but `GradeClass` is closely tied to `GPA`, and the dataset has less storytelling depth.

## Baseline for Chosen Dataset

Chosen baseline: Random Forest on OULAD early-30-day engineered features, excluding `id_student`.

- Accuracy: 0.538
- Weighted precision: 0.519
- Weighted recall: 0.538
- Weighted F1: 0.494
- Test rows: 6,519

Per-class pattern: the model detects `Pass` and `Withdrawn` much better than `Distinction`. That is useful for the project because it motivates class weighting, threshold tuning, richer engagement features, and a high-aptitude recall discussion.

Generated plots:

- `artifacts/oulad_class_distribution.png`
- `artifacts/oulad_numeric_correlation_heatmap.png`

## Recommended Split Strategy

Use a stratified 70/15/15 train/validation/test split on `final_result`.

For a stronger experimental design, also run a robustness split by holding out one `code_presentation` or module-presentation combination as a temporal/generalization test. Keep this as an advanced experiment after the main stratified baseline.

## Required Preprocessing

- Drop identifiers such as `id_student` from model inputs.
- Impute `imd_band` with an explicit `Unknown` category.
- Impute `date_registration` with median or add a missingness flag.
- One-hot encode categorical variables for Scikit-learn.
- Scale numerical variables for logistic regression, SVM, and neural networks.
- Use class weights or focal loss/oversampling because `Distinction` is underrepresented.
- Keep leakage discipline: avoid `date_unregistration`; use assessment scores only in clearly labeled later-window experiments.

## Feature Engineering Ideas

- Compare early engagement windows: days 0-7, 0-14, 0-30, and 0-60.
- Add VLE activity-type clicks by joining `studentVle` to `vle`.
- Add engagement consistency: active weeks, max inactivity gap, clicks per week slope.
- Add early assessment features only when the experiment is framed as "after first assessment".
- Add module-relative features: percentile rank of early clicks within each module presentation.
- Collapse `final_result` into binary high-aptitude (`Distinction` vs rest) as a secondary analysis while keeping multi-class as the main task.
- Test fairness-style slices by `imd_band`, region, disability, and age band.

## Suggested 7+ Experiments

1. Logistic Regression baseline with one-hot encoding and scaling.
2. Random Forest baseline.
3. Gradient Boosting or HistGradientBoosting.
4. SVM or calibrated linear SVM on sampled/encoded features.
5. TensorFlow Sequential API dense network using `tf.data`.
6. TensorFlow Functional API model with separate numeric and categorical input branches.
7. Class-weighted version of the best traditional and deep model.
8. Early-window comparison: 7 vs 14 vs 30 vs 60 days.
9. Feature ablation: demographics only vs engagement only vs combined.

## Reproducibility Files

- Audit script: `dataset_selection_audit.py`
- Full machine-readable results: `artifacts/dataset_audit_summary.json`
- Engineered chosen dataset: `data/processed/oulad_early30.csv`
