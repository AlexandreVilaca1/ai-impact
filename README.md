# AI Impact on Jobs by 2030

Which jobs are most at risk of automation by 2030, and what do they have in common?
This project explores a dataset of 3,000 job records with four techniques: exploratory data
analysis, K-Means clustering, classification (KNN and Logistic Regression) and association
rule mining (Apriori).

Academic project for the Artificial Intelligence course of the BSc in Computer Systems
Engineering at IPCA (Polytechnic Institute of Cávado and Ave), 2025.

## Dataset

`AI_Impact_on_Jobs_2030.csv`: 3,000 rows, 18 columns, 20 job titles, no missing values
or duplicates.

| Column | Description |
|---|---|
| `Job_Title` | Job role (e.g. Software Engineer, Teacher, Retail Worker) |
| `Average_Salary` | Average yearly salary |
| `Years_Experience` | Years of experience |
| `Education_Level` | High School, Bachelor's, Master's or PhD |
| `AI_Exposure_Index` | How exposed the job is to AI (0–1) |
| `Tech_Growth_Factor` | Technological growth factor |
| `Automation_Probability_2030` | Probability that the job is automated by 2030 (0–1) |
| `Risk_Category` | Low, Medium or High automation risk |
| `Skill_1` … `Skill_10` | Skill indicators (0–1) |

## Notebooks

| Notebook | Technique | What it does |
|---|---|---|
| [`EDA.ipynb`](EDA.ipynb) | Exploratory data analysis | Data quality checks, distributions, relationships between salary, AI exposure and automation probability, correlations, comparisons by risk and education, outlier detection (IQR and z-score) |
| [`clustering.ipynb`](clustering.ipynb) | K-Means | Groups jobs by salary, education and automation probability. Picks *k* with the elbow method, silhouette score and Davies-Bouldin index, and plots the clusters with PCA |
| [`automatic_classification.ipynb`](automatic_classification.ipynb) | KNN, Logistic Regression | Predicts each job's risk category (Low / Medium / High) and compares the two models |
| [`association_rules_analysis.ipynb`](association_rules_analysis.ipynb) | Apriori | Discretises the attributes and mines "IF … THEN risk" rules, tuning the support and confidence thresholds |

## Main results

**Exploratory analysis**
- Risk categories: 1,521 Medium, 740 High and 739 Low.
- Highest average automation probability: Retail Worker, Security Guard and Construction
  Worker (0.83). Lowest: AI Engineer, Research Scientist, Nurse, Teacher and Doctor (all
  0.17–0.18).

**Clustering (K-Means)**
- *k* = 8 clusters over all 3,000 jobs, chosen from *k* = 2–10. The highest silhouette
  score is at *k* = 8 (0.297) and the Davies-Bouldin index is almost the same as at
  *k* = 7 (0.977 vs 0.971). *k* = 8 is also the most interpretable: jobs fall along three
  axes with two levels each (salary, education, automation risk), so each cluster is one
  of the 2 × 2 × 2 combinations.
- **Education is not a safety net**: two clusters made up mostly of Master's and PhD holders
  (~24% of jobs) have a 70–74% average automation probability.
- **A high salary is not either**: two clusters earning ~$116–121k on average still have a
  70–71% automation probability.
- **Salary and education do not decide the risk**: every salary × education combination
  appears twice, once with low risk (28–31%) and once with high risk (70–74%). The
  high-risk and low-risk clusters each cover about half of the jobs.

**Classification**

80/20 stratified train/test split, features standardised, education encoded as an ordinal
scale (1–4).

| Model | Accuracy |
|---|---|
| KNN (*k* = 5) | 72.7% |
| KNN tuned (*k* = 8, distance-weighted) | 76.2% |
| Logistic Regression | 99.0% |

**Association rules (Apriori)**
- Support and confidence tested over a grid (support 0.04–0.10, confidence 0.3–0.9).
  With minimum support 0.04, Apriori finds 388 frequent itemsets.
- Strongest high-risk rule: *IF Senior level AND high automation probability THEN high
  risk* (confidence 81%, lift 3.3).

## A note on the results

`Risk_Category` is a binned version of `Automation_Probability_2030`: Low is 0.05–0.30,
Medium 0.31–0.70 and High 0.71–0.95. Both the classifiers and the association rules use
the automation probability as an input, so:

- Logistic Regression's 99% accuracy mostly comes from learning those two cut-off points.
- The strongest association rules include "high automation probability", which by
  definition means high risk.

A more informative next step is to drop `Automation_Probability_2030` from the inputs and
measure how well salary, experience, education, AI exposure and skills alone predict the
risk category.

## Running the notebooks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
jupyter notebook
```

On Windows, use `python` instead of `python3` and `.venv\Scripts\activate` instead of the
`source` line.

Each notebook loads `AI_Impact_on_Jobs_2030.csv` from the project folder and runs from top
to bottom.

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Mlxtend · Jupyter

## Team

João Paulo Carvalho · Pedro Taveira · Alexandre Vilaça · Licínio Cunha
