# Task 3: Customer Churn Prediction (Bank Customers)

> **Internship:** Data Science & Analytics — DevelopersHub Corporation  
> **Tools:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Google Colab

---

## Task Objective

Customer churn — when a customer stops using a bank's services and closes their account — is a major and costly problem in the banking industry. Acquiring a new customer costs significantly more than retaining an existing one.

The objective of this task is to build a **classification model** that identifies which bank customers are **likely to leave (churn)** based on their demographic profile and banking behavior. This allows the bank to proactively take retention actions before the customer leaves.

**Target Variable:** `Exited` — 1 (Churned / Left the bank) or 0 (Retained / Stayed)

---

## Approach

### 1. Data Loading
- Loaded the Churn Modelling Dataset from a public CSV mirror (auto-downloads in the notebook)
- Manual Kaggle upload option also provided in the notebook

### 2. Data Cleaning & Preparation
- Dropped non-predictive identifier columns: `RowNumber`, `CustomerId`, `Surname`
- Confirmed no missing values in the dataset
- Retained all remaining 10 features for modeling

### 3. Categorical Feature Encoding
Two encoding strategies were applied depending on the feature type:

| Feature | Encoding Method | Reason |
|---------|----------------|--------|
| `Gender` | Label Encoding (Female=0, Male=1) | Binary category — 2 values only |
| `Geography` | One-Hot Encoding | Multi-class (France, Germany, Spain) — avoids ordinal assumption |

### 4. Exploratory Data Analysis (EDA)
Visualizations created to explore churn patterns:

| Chart | Insight Sought |
|-------|---------------|
| Pie chart — Churn distribution | Overall churn rate |
| Bar chart — Churn by Gender | Does gender affect churn? |
| Histogram — Age by Churn | Which age groups churn most? |
| Scatter plot — Balance vs Salary | Financial profile of churners |
| Box plots — CreditScore, Age, Balance, NumOfProducts | Feature spread by churn |
| Correlation Heatmap | Relationships between all features |

### 5. Model Training
- Split data: **80% training / 20% testing** with stratified sampling
- Applied `StandardScaler` for feature normalization
- Trained two ensemble models:
  - **Random Forest Classifier** (100 trees, max_depth=8)
  - **Gradient Boosting Classifier** (100 estimators, learning_rate=0.1)

### 6. Model Evaluation
- Accuracy Score
- Confusion Matrix (side-by-side for both models)
- Classification Report (Precision, Recall, F1-score per class)

### 7. Feature Importance Analysis
- Extracted feature importances from the Random Forest model
- Plotted a horizontal bar chart highlighting the top contributing features

---

## Results & Insights

### Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | ~86–87% |
| Gradient Boosting | ~86–87% |

Both models perform similarly. Random Forest is preferred for its interpretability and faster training.

### Key Findings

1. **Age is the most influential churn driver** — older customers are significantly more likely to leave. The age group 40–60 shows the highest churn rate. Banks should prioritize retention efforts for this segment.

2. **Geography matters significantly** — customers from **Germany churn at nearly double the rate** of customers from France and Spain. This suggests region-specific issues such as local competition or service quality.

3. **Number of Products has a non-linear effect** — customers with 2 products have the lowest churn rate and are the most loyal. Customers with only 1 product or 4 products have very high churn rates.

4. **Account Balance pattern** — customers with a zero balance are very likely to churn. Customers with a mid-to-high balance but low activity are also at risk.

5. **Gender** — female customers churn at a slightly higher rate than male customers, though gender alone is not a strong predictor.

6. **Credit Score** — customers who churn tend to have slightly lower credit scores, but the difference is not dramatic.

### Business Recommendations
- Launch **loyalty programs** for 40–60 age group customers
- Investigate why **Germany** has high churn — possibly pricing or competitor offers
- Encourage single-product customers to **adopt a second product** (cross-selling) to increase loyalty
- Flag customers with **sudden balance drops** as early churn signals

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — Churn Modelling Dataset |
| Kaggle Link | https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling |
| Rows | 10,000 |
| Columns | 14 (10 features after preprocessing) |
| Target | `Exited` (1 = Churned, 0 = Retained) |
| Missing Values | None |
| Class Imbalance | ~20% churned, ~80% retained |
