# Customer Churn Prediction using Logistic Regression

This project focuses on building an **industry-acceptable customer churn prediction model** using Logistic Regression.  
The goal is to predict whether a customer is likely to churn based on historical data while following best practices used in real-world machine learning systems.

---

## 📌 Problem Statement

Customer churn refers to customers leaving a company’s service.  
Predicting churn in advance allows businesses to take preventive actions such as discounts, offers, or personalized engagement.

- **Problem Type:** Binary Classification  
- **Target Variable:** `Churn` (Yes / No)

---

## 🧠 Approach & Methodology

The project follows a **production-style machine learning workflow**:

1. Data loading and inspection
2. Feature–target separation
3. Stratified train-test split to handle class imbalance
4. Separate preprocessing for:
   - Numerical features
   - Categorical features
5. Pipeline-based model training to avoid data leakage
6. Proper evaluation using classification metrics

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

## 🔄 Data Preprocessing

- **Missing Values**
  - Numerical: Mean imputation
  - Categorical: Constant value (`Unknown`)
- **Feature Scaling**
  - StandardScaler for numerical features
- **Encoding**
  - OneHotEncoder for categorical features
- **Pipeline & ColumnTransformer**
  - Ensures clean, reusable, and leakage-free preprocessing

---

## 🤖 Model Used

### Logistic Regression
- Chosen as a **baseline, interpretable, and business-trusted model**
- Handles class imbalance using:
  ```python
  class_weight="balanced"
