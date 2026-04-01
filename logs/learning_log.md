## Learning Log: Customer Churn Prediction Using Machine Learning

### Topic
Customer churn prediction for a telecom company using a supervised machine learning pipeline.

### Why this project matters
This is a strong project because it shows the full workflow, not just model training. It covers:

- business problem framing
- dataset understanding
- categorical encoding
- train/test split
- class imbalance handling
- baseline model comparison
- evaluation tradeoffs
- saving preprocessing + model artifacts
- making predictions on new unseen data

### Problem statement
The goal is to predict whether a telecom customer will churn, meaning whether they will leave or cancel the service. The model uses customer information such as demographics, subscribed services, tenure, and billing-related features to predict whether the customer will stay or leave.

### Dataset refresher
Key reminders from the project:

- target column: `Churn`
- important numeric features: `tenure`, `MonthlyCharges`, `TotalCharges`
- many other columns are categorical
- dataset size mentioned in the walkthrough: about 7,043 rows and 21 columns

This is basically a classic tabular binary classification problem.

### End-to-end workflow used in the video
1. collect/load the CSV dataset
2. inspect shape, columns, and data types
3. do EDA to understand what preprocessing is needed
4. encode categorical features
5. split features and target
6. split into training and test sets
7. fix class imbalance on the training set using SMOTE
8. train multiple baseline models
9. compare them with 5-fold cross-validation
10. choose the strongest baseline model
11. evaluate on the held-out test set
12. save the model and encoders
13. build a prediction flow for new customer input

### Main technical takeaways

#### 1. Start by framing churn clearly
Churn means a user leaves the telecom service. In an interview, I should describe this as a business problem before calling it an ML problem.

Good explanation:
This is a binary classification problem where the goal is to identify customers likely to leave so the business can take preventive retention actions.

#### 2. EDA is not optional
One of the strongest ideas in the walkthrough is that EDA is not just for plotting graphs. It helps decide:

- what preprocessing is needed
- which columns are categorical vs numeric
- whether imbalance exists in the target
- which model families make sense

Interview takeaway:
I should explain EDA as a decision-making step, not just a visualization step.

#### 3. The dataset is heavily categorical
The project works with many object/categorical columns, so preprocessing is a core part of the pipeline.

The instructor uses label encoding for categorical columns and automates it with a loop instead of encoding columns manually one by one.

Why this matters:
- scalable for many categorical columns
- cleaner code
- reproducible preprocessing

Important interview line:
I used a loop to encode categorical columns consistently and stored the encoders by column name for reuse during inference.

#### 4. Save the encoders, not just the model
This was one of the most practical parts of the video.

The project stores label encoders in a dictionary and saves them as a pickle file. That matters because the exact same category-to-number mapping must be reused when predicting on new data.

Why this matters in interviews:
If I only save the model but not the preprocessing objects, the production prediction pipeline may break or silently produce wrong inputs.

#### 5. Split first, then handle imbalance
A key best practice from the walkthrough:

- first split data into training and test sets
- then apply SMOTE only on the training data

That avoids contaminating the test set.

This is an excellent interview point because it shows I understand leakage.

#### 6. SMOTE was used to balance the training data
The walkthrough uses SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance.

Training set before SMOTE:
- class 0: about 4,138
- class 1: about 1,496

Training set after SMOTE:
- class 0: 4,138
- class 1: 4,138

Important interview explanation:
I used SMOTE only on the training set so the model could learn from a more balanced target distribution while still being evaluated on an untouched test set.

#### 7. Model comparison was done with cross-validation
Instead of trusting just one train/test split accuracy, the walkthrough compares three tree-based models using 5-fold cross-validation:

- Decision Tree
- Random Forest
- XGBoost

Approximate cross-validation accuracy mentioned in the video:
- Decision Tree: ~0.78
- Random Forest: ~0.84
- XGBoost: ~0.83

This is a very good interview habit to mention.

Better interview phrasing:
I compared baseline models using 5-fold cross-validation because a single split can be noisy. Cross-validation gives a more reliable estimate of performance.

#### 8. Why tree-based models were chosen
The instructor says tree-based models are a good fit here because:

- they are robust
- they are less sensitive to outliers
- they do not require standardization

He also notes that if I use models such as logistic regression or SVM, I should scale numeric columns like tenure, monthly charges, and total charges.

Interview takeaway:
I should justify model choice based on data type and preprocessing needs, not just say “I used Random Forest because it works well.”

#### 9. Random Forest was used as the final saved baseline
Since Random Forest had the strongest baseline cross-validation result, it was trained and used as the saved model artifact for inference.

This is a simple but strong interview story:
- compare reasonable baselines
- select the best one using evidence
- persist the chosen model for reuse

#### 10. Accuracy alone can be misleading on imbalanced data
On the held-out test set, the walkthrough reports around 78% accuracy, but the important lesson is that accuracy is not the best metric when the evaluation set is imbalanced.

Test-set class counts mentioned in the video were roughly:
- class 0: 1,036
- class 1: 373

The instructor explicitly points out that precision, recall, and the classification report matter more than raw accuracy in this setting.

This is one of the best interview lessons from the whole video.

Interview phrasing:
Because the test data was still imbalanced, I did not rely on accuracy alone. I also checked the confusion matrix and class-wise precision/recall to understand minority-class performance.

#### 11. Cross-validation can still show fold instability
Another subtle point: some folds performed worse than others. The instructor suggests stratified k-fold as a possible improvement.

Why that matters:
It shows awareness that even cross-validation can be unstable when class distribution differs across folds.

Strong interview line:
I noticed fold-to-fold variation, so a next step would be stratified cross-validation to preserve class proportions more consistently.

#### 12. Suggested next improvements
The video intentionally leaves room for follow-up improvements. The instructor mentions ideas such as:

- hyperparameter tuning
- stratified k-fold
- checking train vs test metrics for overfitting
- trying simpler models like logistic regression with scaling

This is useful in interviews because it shows that a project is not “done” just because a model runs.

### Production / deployment mindset from the walkthrough
The project does not stop at training.

It also saves:
- the trained model
- the feature names
- the encoders

Then it demonstrates how to:
- load the saved model
- create a new input record as a dictionary
- convert it into a DataFrame
- transform categorical values with the saved encoders
- generate both a class prediction and a probability using `predict_proba`

That is a very interview-friendly end-to-end story.

### What I should be ready to explain in an interview

#### Why did you apply SMOTE after the train/test split?
Because applying SMOTE before the split would leak synthetic information into the test set and make evaluation less trustworthy.

#### Why compare multiple models instead of using one immediately?
Because model performance depends on the dataset. Comparing a few strong baselines gives evidence for model selection instead of guessing.

#### Why save label encoders separately?
Because new input data must be transformed exactly the same way as training data. Otherwise the model sees inconsistent numeric representations.

#### Why are tree models convenient here?
Because they handle tabular data well, can work with label-encoded features, are relatively robust, and do not require feature scaling.

#### Why is accuracy not enough here?
Because churn prediction is imbalanced. A model can get a decent accuracy while still doing poorly on the customers most likely to churn.

#### What would you improve next?
I would try hyperparameter tuning, stratified cross-validation, stronger evaluation focused on the minority class, and a comparison with simpler or more regularized models.

### Interview-ready summary
I built a telecom customer churn prediction pipeline using tabular customer data. The main technical challenges were many categorical features and class imbalance. I encoded the categorical columns, split the data into train and test sets, applied SMOTE only to the training data, then compared Decision Tree, Random Forest, and XGBoost with 5-fold cross-validation. Random Forest gave the best baseline cross-validation result, so I used it as the saved model. I evaluated it on a held-out imbalanced test set using more than just accuracy, then saved both the model and encoders so I could make consistent predictions on new customer records.

### Key phrases to remember
- This is a supervised binary classification problem.
- Churn prediction is useful because retention is usually cheaper than reacquisition.
- I handled many categorical columns with saved label encoders.
- I applied SMOTE only on the training set to avoid leakage.
- I used cross-validation to compare baseline models more reliably.
- Accuracy alone is not enough for imbalanced churn data.
- I saved the full inference pipeline, not only the model.

### My biggest takeaways
- Preprocessing decisions are part of the model story.
- Leakage prevention matters.
- Model comparison should be evidence-based.
- Evaluation must reflect class imbalance.
- A project becomes much stronger when it includes inference and artifact saving.

### What I still want to revise
- exact difference between label encoding and one-hot encoding
- when SMOTE helps and when it can hurt
- how stratified k-fold differs from normal k-fold
- how to tune Random Forest and XGBoost properly
- which metric matters most for a real churn-retention use case

### 30-second refresher before an interview
This project was an end-to-end telecom churn classification pipeline. I used the Telco churn dataset, encoded many categorical features, split train and test data, handled class imbalance with SMOTE on the training set only, compared Decision Tree, Random Forest, and XGBoost with 5-fold cross-validation, selected Random Forest as the strongest baseline, evaluated beyond accuracy because the test set was imbalanced, and saved both the model and encoders to support consistent predictions on new customer i
::contentReference[oaicite:2]{index=2}
nputs.