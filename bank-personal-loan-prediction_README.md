# 🏦 Bank Personal Loan Prediction

A binary classification project predicting which bank customers are likely
to accept a personal loan offer, with a strong focus on handling severe
class imbalance and comparing multiple classifiers on business-relevant
metrics.

---

## 🎯 Objective

Build a model that identifies which customers are most likely to accept a
personal loan offer, so the bank can target marketing more effectively and
avoid wasted outreach — while explicitly correcting for a ~9:1 class
imbalance in the historical data (only ~480 of 5,000 customers accepted).

## 📊 Dataset

- **Source:** Bank Personal Loan Modelling dataset (`Bank_Personal_Loan_Modelling.csv`)
- **Size:** ~5,000 customers with demographic, financial, and account
  features (age, income, experience, credit-card spend, education,
  mortgage, family size, existing accounts, etc.)

> Note: link to the original dataset source instead of committing the raw
> CSV, unless the license allows redistribution.

## 🔍 Exploratory Data Analysis — Key Findings

- Detected and corrected invalid negative values in the `Experience` column.
- Confirmed severe class imbalance in the target (`Personal Loan`).
- Found `Income` and `CCAvg` (average credit card spend) to be the strongest
  behavioral predictors — customers who received loan offers had
  meaningfully higher income and more than double the CCAvg of those who
  didn't, across every education level.
- Showed that `Experience` and `Mortgage` are weak predictors on their own,
  despite `Experience` initially seeming important.
- Mapped the customer base geographically and confirmed a concentration in
  the San Francisco and Los Angeles metro areas.
- Applied log transformations to `Income` and `CCAvg` to reduce skew ahead
  of modeling.

## ⚖️ Handling Class Imbalance

Multiple resampling strategies were tested; **SMOTE (Synthetic Minority
Oversampling)** produced the best overall balance of precision and recall
and was carried forward into final model training.

## 🤖 Modeling & Results

Three classifiers were trained on the SMOTE-balanced data and evaluated on
the untouched test set for the positive ("loan accepted") class:

| Model                                                 | Precision | Recall | F1-Score    |
| ----------------------------------------------------- | --------- | ------ | ----------- |
| Logistic Regression                                   | 0.51      | 0.93   | 0.66        |
| **K-Nearest Neighbors (k=2, GridSearchCV-validated)** | **0.73**  | 0.69   | **0.71** 🏆 |
| Gaussian Naive Bayes                                  | 0.43      | 0.84   | 0.57        |

**Champion model:** **K-Nearest Neighbors (k=2)** — the best F1-Score and,
critically, the highest precision, making it the most reliable and
lowest-risk model for targeted loan marketing.

## ✅ Conclusion

By combining careful feature engineering, transparent handling of class
imbalance via SMOTE, and hyperparameter validation (GridSearchCV), the final
KNN model offers the bank a dependable way to prioritize loan offers toward
customers who are genuinely likely to accept — reducing wasted marketing
spend while capturing the majority of real opportunities.

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn (SMOTE)` ·
`matplotlib` · `seaborn`

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook bank_personal_loan_prediction.ipynb
```

## 📄 License

MIT (or your preferred license) — add a `LICENSE` file to the repo.

---

> ⚠️ **Before publishing:** two markdown cells in the original notebook are
> written in Persian (around the "Experience" and "under-26" analysis
> sections). Translate these to English so the whole notebook reads
> consistently for an international audience.
