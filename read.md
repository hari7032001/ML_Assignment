# Machine Learning Assignment 2  
## Bank Marketing Classification

---

## a. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a customer will subscribe to a term deposit based on the Bank Marketing dataset.

The task is a binary classification problem where the target variable indicates whether the client subscribed to a term deposit (yes/no).

---

## b. Dataset Description

Dataset Used: Bank Marketing Dataset (bank-full.csv)

Source: UCI Machine Learning Repository

Minimum Feature Size: 16 features  
Minimum Instance Size: 45,211 records  

The dataset contains demographic and campaign-related information of customers contacted during direct marketing campaigns conducted by a Portuguese banking institution.

Target Variable:
- y = yes (1)
- y = no (0)

The dataset contains both categorical and numerical features such as:
- age
- job
- marital status
- education
- balance
- housing loan
- previous campaign outcome
- etc.

Preprocessing Steps:
- Converted target labels to binary (yes = 1, no = 0)
- Encoded categorical variables using Label Encoding
- Standardized features using StandardScaler
- Split dataset into 80% training and 20% testing data using stratified sampling

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.8914 | 0.8726 | 0.5945 | 0.2259 | 0.3274 | 0.3205 |
| Decision Tree | 0.8778 | 0.7061 | 0.4780 | 0.4820 | 0.4800 | 0.4108 |
| KNN | 0.8923 | 0.8089 | 0.5717 | 0.3166 | 0.4075 | 0.3724 |
| Naive Bayes | 0.8380 | 0.8127 | 0.3554 | 0.4726 | 0.4057 | 0.3183 |
| Random Forest (Ensemble) | 0.9064 | 0.9250 | 0.6563 | 0.4206 | 0.5127 | 0.4777 |
| XGBoost (Ensemble) | 0.9058 | 0.9267 | 0.6281 | 0.4773 | 0.5424 | 0.4968 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved high accuracy but low recall, indicating difficulty in identifying positive class instances effectively. |
| Decision Tree | Balanced precision and recall but relatively lower AUC compared to ensemble models. |
| KNN | Moderate performance; slightly better recall than Logistic Regression but still limited in capturing minority class. |
| Naive Bayes | Lower accuracy but relatively better recall compared to Logistic Regression, due to probabilistic assumptions. |
| Random Forest (Ensemble) | Achieved high accuracy and strong AUC; demonstrated better generalization and balanced performance. |
| XGBoost (Ensemble) | Best overall model based on AUC, F1 score, and MCC; handled complex feature interactions effectively. |

---

