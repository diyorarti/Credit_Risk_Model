# Credit Risk Modeling in Python

## Project Overview

This project is a complete **Credit Risk Modeling** case study built in Python.  
The main goal of the project is to estimate the **Probability of Default (PD)** of loan applicants and use the model output for credit decision-making.

The project follows a traditional credit risk modeling workflow:

1. Data preprocessing
2. Feature engineering
3. Weight of Evidence (WoE) and Information Value (IV) analysis
4. Logistic Regression model training
5. Model validation
6. Scorecard creation
7. Credit score calculation
8. Conversion from credit score to Probability of Default
9. Setting cut-off thresholds for lending decisions

This project was completed as part of my learning journey in credit risk modeling, data science, and financial machine learning.

---

## Business Problem

Banks and financial institutions need to decide whether a loan applicant is likely to repay their loan or default.

The key business question is:

> Can we estimate the probability that a borrower will default and use that information to support loan approval decisions?

In credit risk modeling, this is usually called a **PD model**, where PD means **Probability of Default**.

A borrower with a high PD is more risky, while a borrower with a low PD is safer.

---

## Project Objective

The objective of this project is to build a model that can:

- Predict whether a borrower is likely to be good or bad
- Estimate the probability of default
- Evaluate the model using credit risk metrics
- Convert model outputs into a scorecard
- Use score cut-offs to support lending decisions

---

## Dataset

The project uses historical loan data containing borrower information, loan characteristics, credit history, and loan status.

The target variable is created from the loan status column.

In this project:

- `1` means **good borrower**
- `0` means **bad borrower**

Bad borrowers include loans with statuses such as:

- Charged Off
- Default
- Late (31-120 days)
- Does not meet the credit policy. Status: Charged Off

Good borrowers include loans that were successfully paid or did not show serious default behavior.

---

## Project Structure

```text
Credit-Risk-Modeling/
│
├── data_processing_part_1.ipynb
├── data_processing_part_2.ipynb
├── 1_logistic_regression_model_training.ipynb
├── 2_logistic_regression_model_validation.ipynb
├── 3_applying_model_for_decision_making.ipynb
│
├── data/
│   ├── inputs_train.csv
│   ├── inputs_test.csv
│   ├── target_train.csv
│   └── target_test.csv
│
├── trained_models/
│   ├── logistic_regression_model
│   ├── LG_Model_pvalues_summary_table
│   └── y_hat_test_data_predicted_probabilities.csv
│
└── README.md
```

---

## Notebook 1: Data Processing Part 1

The first notebook focuses on initial data cleaning and target variable creation.

Main steps:

- Load the raw dataset
- Explore the dataset structure
- Clean important variables
- Convert date variables
- Handle missing values
- Create the target variable `good_bad`
- Prepare the dataset for further preprocessing

Important variables processed in this stage include:

- `term`
- `emp_length`
- `issue_d`
- `earliest_cr_line`
- `loan_status`

The target variable is created using loan status:

```python
data['good_bad'] = np.where(
    data['loan_status'].isin([
        'Charged Off',
        'Default',
        'Does not meet the credit policy. Status:Charged Off',
        'Late (31-120 days)'
    ]),
    0,
    1
)
```

---

## Notebook 2: Data Processing Part 2

The second preprocessing notebook focuses on feature engineering and credit-risk-style variable transformation.

Main steps:

- Create dummy variables
- Group categorical variables
- Analyze Weight of Evidence (WoE)
- Analyze Information Value (IV)
- Create grouped/binned variables
- Prepare final input features
- Split data into training and testing sets
- Save processed datasets

### Weight of Evidence

Weight of Evidence is used to measure the relationship between a feature category and the target variable.

It helps to understand whether a group is associated with good or bad borrowers.

### Information Value

Information Value measures the predictive power of a variable.

In credit risk modeling, IV is often used to decide whether a variable is useful for modeling.

General interpretation:

| IV Value | Predictive Power |
|---|---|
| Less than 0.02 | Not useful |
| 0.02 - 0.1 | Weak |
| 0.1 - 0.3 | Medium |
| 0.3 - 0.5 | Strong |
| More than 0.5 | Very strong / suspicious |

---

## Notebook 3: Logistic Regression Model Training

The third notebook trains the PD model using Logistic Regression.

Logistic Regression is commonly used in credit risk modeling because it is:

- Interpretable
- Easy to explain
- Suitable for binary classification
- Commonly used in scorecard development

Main steps:

- Load preprocessed training data
- Select final model features
- Remove reference categories
- Train Logistic Regression
- Extract model coefficients
- Calculate approximate p-values
- Create a model summary table
- Remove statistically weaker variables
- Save the trained model
- Save the summary table

The model predicts the probability that a borrower belongs to class `1`, which means a **good borrower**.

Therefore:

```text
Probability of Default = 1 - Probability of Being Good
```

---

## Notebook 4: Model Validation

The fourth notebook validates the trained Logistic Regression model using test data.

Main steps:

- Load test data
- Load the trained model
- Predict probabilities
- Create actual vs predicted table
- Apply classification thresholds
- Build confusion matrix
- Plot ROC curve
- Calculate AUC
- Calculate Gini
- Calculate KS statistic

### AUC

AUC measures how well the model separates good and bad borrowers.

A higher AUC means better ranking ability.

### Gini

Gini is calculated from AUC:

```text
Gini = 2 × AUC - 1
```

Gini is commonly used in credit risk modeling to measure model discriminatory power.

### KS Statistic

The Kolmogorov-Smirnov statistic measures the maximum distance between cumulative good borrowers and cumulative bad borrowers.

It shows how well the model separates risky and safe borrowers.

---

## Notebook 5: Applying the Model for Decision Making

The final notebook applies the PD model to real credit decision-making.

Main steps:

- Load the model summary table
- Add reference categories back to the scorecard
- Assign coefficient `0` to reference categories
- Create a scorecard
- Convert model coefficients into score points
- Calculate credit scores for borrowers
- Convert credit scores back into PD values
- Create approval/rejection cut-offs

This part is important because it turns the machine learning model into a practical decision-making tool.

The workflow is:

```text
Model coefficients
        ↓
Scorecard points
        ↓
Borrower credit score
        ↓
Probability of Default
        ↓
Credit decision
```

Example decision logic:

```text
High score → Low PD → Safer borrower → Approve
Low score → High PD → Riskier borrower → Reject or review
```

---

## Credit Scorecard

A scorecard converts the Logistic Regression model into a point-based system.

Instead of only using model coefficients or predicted probabilities, the scorecard gives each borrower a credit score.

In this project, the score range is:

```python
min_score = 300
max_score = 850
```

This means:

- Minimum possible score: `300`
- Maximum possible score: `850`

A higher score means the borrower is safer.  
A lower score means the borrower is riskier.

---

## Model Decision-Making

After calculating the credit score and PD, cut-offs can be used to make decisions.

Example:

| Credit Score | Risk Level | Decision |
|---|---|---|
| 700+ | Low risk | Approve |
| 600 - 699 | Medium risk | Manual review |
| Below 600 | High risk | Reject |

The exact cut-off depends on the bank’s risk appetite and business strategy.

A stricter cut-off reduces default risk but may reject many good customers.  
A more flexible cut-off increases approvals but may increase credit losses.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle
- Jupyter Notebook

---

## Main Python Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import pickle
```

---

## Key Credit Risk Concepts Covered

This project covers several important credit risk modeling concepts:

- Probability of Default (PD)
- Good borrower vs bad borrower classification
- Logistic Regression for PD modeling
- Weight of Evidence (WoE)
- Information Value (IV)
- Reference categories
- Scorecard development
- Credit score calculation
- Gini coefficient
- AUC
- KS statistic
- Cut-off selection
- Model-based decision-making

---

## Results and Learnings

Through this project, I learned how a credit risk model is built from raw data to final business decision-making.

The most important learning points were:

- How to define good and bad borrowers
- How to clean and transform loan data
- How to use WoE and IV for feature analysis
- Why Logistic Regression is useful in credit risk modeling
- How to evaluate a PD model using AUC, Gini, and KS
- How to convert model coefficients into a scorecard
- How credit scores can be used for loan approval decisions

---

## Known Limitations

This project was completed as a learning and portfolio project.  
There are several areas that could be improved in a production-level credit risk model:

- WoE binning should ideally be fitted only on training data to avoid data leakage
- More advanced validation could be added
- Population Stability Index (PSI) could be used for monitoring
- Reject inference is not included
- Model calibration could be improved
- More business rules could be added to the final decision strategy
- The target definition could be adjusted based on formal banking default definitions

These limitations are part of the learning process and show areas for future improvement.

---

## Future Improvements

Possible improvements for the project:

- Add Population Stability Index (PSI)
- Add model monitoring
- Add score distribution analysis
- Add approval rate and bad rate analysis by cut-off
- Add expected loss calculation using PD, LGD, and EAD
- Compare Logistic Regression with other models
- Build a simple dashboard for credit decision-making
- Create a Streamlit application for borrower risk prediction
- Improve code modularity using Python scripts
- Add unit tests for preprocessing functions

---

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/credit-risk-modeling.git
```

2. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Open Jupyter Notebook:

```bash
jupyter notebook
```

4. Run the notebooks in this order:

```text
1. data_processing_part_1.ipynb
2. data_processing_part_2.ipynb
3. 1_logistic_regression_model_training.ipynb
4. 2_logistic_regression_model_validation.ipynb
5. 3_applying_model_for_decision_making.ipynb
```

---

## Why This Project Matters

This project is important because it connects machine learning with real financial decision-making.

Many machine learning projects stop after model training and accuracy calculation.  
This project goes further by creating a scorecard and applying the model to credit decisions.

It shows not only technical skills, but also understanding of how models are used in banking and risk management.

---

## Author

**Diyorbek Tirkashov**

Computer Science student based in Poland, focused on Data Analytics, Machine Learning, and Credit Risk Modeling.

GitHub: [https://github.com/diyorarti](https://github.com/diyorarti)  
Kaggle: [https://www.kaggle.com/diyorarti](https://www.kaggle.com/diyorarti)

---

## Project Status

Completed as a portfolio project.

The project demonstrates an end-to-end credit risk modeling workflow from preprocessing to model-based decision-making.
