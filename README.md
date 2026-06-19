# Early Identification of High-Aptitude Students from Online Engagement Behaviour

Final project for *Introduction to Machine Learning* at the African Leadership University.

**Problem.** Talented students in under-resourced schools often go unnoticed because teachers lack the tools and training to spot subject-specific strengths early. This project asks whether observable engagement behaviour, captured in the first 30 days of a course, can identify which students will excel, while there is still time to act.

**Dataset.** The Open University Learning Analytics Dataset (OULAD), a public set of 32,593 student-course records with demographics, assessment results, and over 10 million rows of virtual-learning-environment clickstream activity. The notebook downloads it automatically from the UCI Machine Learning Repository (dataset #349) when the raw files are not present locally.

**Approach.** Eight systematically varied experiments compare traditional machine learning (Scikit-learn) with deep learning (TensorFlow, both the Sequential and Functional APIs, fed through `tf.data`), all restricted to first-30-day features:

1. Logistic regression baseline (class-weighted)
2. Random forest baseline
3. Random forest tuned with `GridSearchCV`
4. Histogram gradient boosting with SMOTE
5. Sequential neural network
6. Sequential network with Dropout and L2
7. Functional two-branch network (behaviour vs. demographics)
8. Binary reformulation: Distinction vs. rest

Every experiment reports a classification report, confusion matrix, ROC curves, and a learning curve. The headline result: the binary Distinction-vs-rest model reaches 0.827 macro AUC, and behavioural features dominate demographic ones.

## Repository structure

```
oulad_summative_notebook.ipynb   Main deliverable — runs top to bottom
oulad_early30.csv                Processed first-30-day modelling table (cache)
experiment_results.csv           Consolidated results from all 8 experiments
DATASET_RECOMMENDATION.md         Why OULAD was chosen over the alternatives
dataset_selection_audit.py        Script behind the dataset comparison
dataset_exploration/             Exploration scripts and generated figures
artifacts/                       Saved best neural-network weights
report/
  report.md                      Source of the written report
  build_pdf.py                   Renders report.md -> PDF
  build_docx.js                  Renders report.md -> Word
  figures/                       Figures used in the report
```

## Reproducing the results

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn matplotlib seaborn tensorflow imbalanced-learn jupyter
jupyter notebook oulad_summative_notebook.ipynb
```

The first code cell checks and installs any missing dependencies, so the notebook also runs as-is on Google Colab. All random seeds are fixed to 42. A full cold run, including the one-time OULAD download, takes roughly 12 minutes on CPU.

## Report and demo

- Written report: `report/summative_report.pdf`
- Demo video: *(link to be added)*
