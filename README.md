# Telco Customer Churn Prediction

## Overview

This project builds an end to end machine learning pipeline to predict customer churn in a telecom company. The goal is to identify customers likely to leave so that the business can take proactive retention actions.

The project focuses not only on model performance but also on building a reproducible and production ready workflow.

## Problem Statement

Customer churn refers to customers canceling their service. This is a supervised binary classification problem where the model predicts whether a customer will churn or not based on historical data.

## Dataset

The dataset contains around 7043 rows and 21 columns.

Target column:
Churn

Important numerical features:
tenure
MonthlyCharges
TotalCharges

The dataset also contains many categorical features related to customer demographics and subscribed services.

## Project Workflow

1. Load and inspect the dataset
2. Perform exploratory data analysis
3. Encode categorical features
4. Split data into training and test sets
5. Handle class imbalance using SMOTE
6. Train multiple baseline models
7. Evaluate models using cross validation
8. Select the best performing model
9. Evaluate on a held out test set
10. Save model and preprocessing artifacts
11. Build a prediction pipeline for new data

## Key Decisions

### Categorical Encoding

Categorical columns were label encoded using a loop. Encoders were stored in a dictionary so they can be reused during inference.

This ensures consistent transformation of new data.

### Train Test Split Before SMOTE

The dataset was split into training and test sets before applying SMOTE.

SMOTE was applied only to the training data to avoid data leakage and preserve the integrity of the test set.

### Handling Class Imbalance

The training data was imbalanced, so SMOTE was used to balance the classes.

Before SMOTE:
Class 0 around 4138
Class 1 around 1496

After SMOTE:
Class 0 4138
Class 1 4138

### Model Comparison

Three models were compared using 5 fold cross validation:

Decision Tree
Random Forest
XGBoost

Approximate cross validation accuracy:
Decision Tree around 0.78
Random Forest around 0.84
XGBoost around 0.83

Random Forest performed best and was selected as the final model.

### Model Choice

Tree based models were chosen because they:
Handle tabular data well
Are robust to outliers
Do not require feature scaling
Work well with encoded categorical data

### Evaluation

Accuracy alone is not sufficient for imbalanced datasets.

The evaluation included:
Confusion matrix
Precision and recall
Classification report

Test accuracy was around 78 percent, but additional metrics were used to better understand performance on the minority class.

### Saving Artifacts

The project saves:
The trained model
Label encoders
Feature structure

This allows consistent and reliable predictions on new data.

## Prediction Pipeline

To make predictions on new customer data:

1. Convert input into a DataFrame
2. Apply saved encoders
3. Use the trained model to generate:
   class prediction
   probability using predict_proba

## Key Learnings

Preprocessing is a critical part of the pipeline
Avoiding data leakage is essential
Model selection should be based on evidence
Evaluation must consider class imbalance
Saving preprocessing objects is necessary for deployment

## Future Improvements

Hyperparameter tuning
Stratified cross validation
Compare with logistic regression and scaling
Improve minority class performance
Analyze overfitting

## Summary

This project implements a complete churn prediction pipeline using supervised machine learning. It handles categorical data, addresses class imbalance using SMOTE, compares multiple models using cross validation, selects Random Forest as the final model, and saves all necessary artifacts for real world prediction.


