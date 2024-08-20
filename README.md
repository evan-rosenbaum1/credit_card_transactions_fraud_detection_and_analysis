# Credit Card Fraud Detection

## Table of Contents
- [Project Overview](#project-overview)
- [Streamlit App](#streamlit-app)
- [Data Summary](#data-summary)
- [Project Summary](#project-summary)
  - [Resampling Techniques](#resampling-techniques)
  - [Classification Models](#classification-models)
  - [Boosting Models](#boosting-models)
  - [Model Evaluation](#model-evaluation)
    - [Top Performing Models (Hypertuned)](#top-performing-models-hypertuned)
    - [Boosted Models (Hypertuned)](#boosted-models-based-on-randomforestclassifier-oversample)
- [Model Score](#model-score)
- [Business Impact](#business-impact)
- [Contact](#contact)

## Project Overview

ABC Bank is experiencing challenges in detecting credit card fraud due to the high volume of transactions processed daily. The existing fraud detection system struggles to identify fraudulent activities, leading to significant financial losses and customer dissatisfaction. The goal of this project is to develop a machine learning model that effectively detects fraudulent transactions, thereby minimizing financial losses, improving customer trust, and enhancing operational efficiency.

## Streamlit App
If you would like to iteract with the model, please click the streamlit button below.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditcardtransactionsfrauddetectionandanalysis-dqsiqchh9v4s6g.streamlit.app)

After answering a questionnaire, you can upload a csv of your credit card transactions. The transactions will be run through the fraud detection model and will be classified as either non-fraudulent or potentially fraudulent. 

The required columns for the csv file are as follows:

- transaction_date
- category
- amt

For the category column, each transaction must be categorized as one of the following:
- home
- health_fitness
- misc_net
- gas_transport
- shopping_net
- personal_care
- misc_pos
- kids_pets
- grocery_pos
- shopping_pos
- travel
- grocery_net
- food_dining
- entertainment

## Data Summary

This project uses the **Credit Card Transaction Records Dataset** from Kaggle, which contains over 1.3 million records of credit card transactions. The dataset includes detailed information such as transaction times, amounts, and personal and merchant data.

For efficient analysis and modeling, the dataset was trimmed to 50,000 rows, with 6,363 transactions labeled as fraudulent.

## Project Summary

### Resampling Techniques

To address the class imbalance in the dataset, resampling techniques were employed:
- Oversampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Borderline SMOTE
- KMeans SMOTE
- ADASYN (Adaptive Synthetic Sampling)
- SMOTE Tomek
- SMOTE ENN

These techniques were used to create a balanced training set, improving the model's ability to detect fraudulent transactions.

### Classification Models

The following classification algorithms were used to provide a baseline performance:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors
- Bagging Classifier
- Random Forest Classifier

These models were evaluated for their effectiveness in fraud detection.

### Boosting Models

To enhance the performance of the best baseline model, boosting models were implemented. The best model, a Random Forest classifier using oversampling, was boosted using the following algorithms:
- Gradient Boosting Classifier
- XGBoost (XGBClassifier)
- LightGBM (LGBMClassifier)

### Model Evaluation

#### Top Performing Models (Hypertuned)

|    | model_name              | sampling_technique | train_prauc_score | test_prauc_score | train_f1_score | test_f1_score |
|----|-------------------------|--------------------|-------------------|------------------|----------------|---------------|
| 39 | RandomForestClassifier   | adasyn             | 0.996153          | 0.971264         | 0.966763       | 0.917740      |
| 35 | RandomForestClassifier   | oversample         | 0.997561          | 0.971186         | 0.973092       | 0.915228      |
| 37 | RandomForestClassifier   | borderline_smote   | 0.993839          | 0.969430         | 0.957976       | 0.916822      |
| 32 | BaggingClassifier        | adasyn             | 0.999787          | 0.968728         | 0.992360       | 0.926082      |
| 29 | BaggingClassifier        | smote              | 0.999215          | 0.968094         | 0.978063       | 0.915418      |
| 30 | BaggingClassifier        | borderline_smote   | 0.999049          | 0.967625         | 0.976085       | 0.913087      |
| 36 | RandomForestClassifier   | smote              | 0.998813          | 0.967090         | 0.978214       | 0.916420      |
| 31 | BaggingClassifier        | kmeans_smote       | 0.998368          | 0.966797         | 0.970995       | 0.920626      |
| 28 | BaggingClassifier        | oversample         | 0.999982          | 0.966126         | 0.977178       | 0.911743      |
| 33 | BaggingClassifier        | smote_tomek        | 0.995300          | 0.964145         | 0.968937       | 0.917071      |
| 40 | RandomForestClassifier   | smote_tomek        | 0.997537          | 0.963731         | 0.981192       | 0.916935      |
| 38 | RandomForestClassifier   | kmeans_smote       | 0.999962          | 0.960225         | 0.995153       | 0.915913      |
| 41 | RandomForestClassifier   | smote_enn          | 0.979806          | 0.955235         | 0.932063       | 0.906896      |
| 34 | BaggingClassifier        | smote_enn          | 0.982690          | 0.952141         | 0.948370       | 0.917603      |
| 14 | DecisionTreeClassifier   | oversample         | 0.982597          | 0.951315         | 0.893427       | 0.868290      |
| 15 | DecisionTreeClassifier   | smote              | 0.982051          | 0.949775         | 0.888159       | 0.867925      |
| 17 | DecisionTreeClassifier   | kmeans_smote       | 0.982907          | 0.949432         | 0.927643       | 0.909671      |
| 19 | DecisionTreeClassifier   | smote_tomek        | 0.978669          | 0.945849         | 0.892853       | 0.875543      |
| 18 | DecisionTreeClassifier   | adasyn             | 0.973869          | 0.940978         | 0.877320       | 0.854948      |
| 16 | DecisionTreeClassifier   | borderline_smote   | 0.977703          | 0.940074         | 0.888708       | 0.865813      |
| 20 | DecisionTreeClassifier   | smote_enn          | 0.943341          | 0.919405         | 0.898677       | 0.884978      |
| 24 | KNeighborsClassifier     | kmeans_smote       | 1.000000          | 0.842965         | 1.000000       | 0.820011      |
| 22 | KNeighborsClassifier     | smote              | 1.000000          | 0.819389         | 1.000000       | 0.796719      |
| 26 | KNeighborsClassifier     | smote_tomek        | 0.993452          | 0.810294         | 0.975779       | 0.794654      |
| 25 | KNeighborsClassifier     | adasyn             | 1.000000          | 0.799919         | 1.000000       | 0.790241      |
| 23 | KNeighborsClassifier     | borderline_smote   | 1.000000          | 0.797335         | 1.000000       | 0.800205      |
| 21 | KNeighborsClassifier     | oversample         | 1.000000          | 0.776127         | 1.000000       | 0.768534      |
| 10 | LogisticRegression       | kmeans_smote       | 0.751337          | 0.774209         | 0.797213       | 0.808223      |
| 3  | BaselineLogisticRegression| kmeans_smote      | 0.751337          | 0.774209         | 0.797213       | 0.808223      |
| 13 | LogisticRegression       | smote_enn          | 0.747920          | 0.770076         | 0.641711       | 0.646757      |
| 6  | BaselineLogisticRegression| smote_enn         | 0.745210          | 0.767682         | 0.635832       | 0.636842      |
| 12 | LogisticRegression       | smote_tomek        | 0.744314          | 0.767291         | 0.629393       | 0.633078      |
| 8  | LogisticRegression       | smote              | 0.743364          | 0.766312         | 0.626251       | 0.633100      |
| 7  | LogisticRegression       | oversample         | 0.741473          | 0.765514         | 0.637936       | 0.647414      |
| 5  | BaselineLogisticRegression| smote_tomek       | 0.740626          | 0.764159         | 0.624524       | 0.630048      |
| 1  | BaselineLogisticRegression| smote             | 0.739587          | 0.763054         | 0.620709       | 0.628250      |
| 0  | BaselineLogisticRegression| oversample        | 0.737881          | 0.762300         | 0.630142       | 0.639030      |
| 9  | LogisticRegression       | borderline_smote   | 0.713455          | 0.738025         | 0.526991       | 0.525521      |
| 11 | LogisticRegression       | adasyn             | 0.707299          | 0.731081         | 0.511714       | 0.509974      |
| 2  | BaselineLogisticRegression| borderline_smote  | 0.697870          | 0.723989         | 0.526908       | 0.526178      |
| 4  | BaselineLogisticRegression| adasyn            | 0.695250          | 0.719708         | 0.512079       | 0.510129      |
| 27 | KNeighborsClassifier     | smote_enn          | 0.834613          | 0.676822         | 0.825196       | 0.708537      |


#### Boosted Models Based on RandomForestClassifier (Oversample)

|    | base_model              | sampling_technique | boosting_model         | train_prauc_score | test_prauc_score | train_f1_score | test_f1_score |
|----|-------------------------|--------------------|------------------------|-------------------|------------------|----------------|---------------|
| 1  | RandomForestClassifier   | adasyn             | LGBMClassifier          | 1.000000          | 0.999435         | 0.999776       | 0.985703      |
| 2  | RandomForestClassifier   | adasyn             | XGBClassifier           | 0.999266          | 0.998899         | 0.984433       | 0.979782      |
| 0  | RandomForestClassifier   | adasyn             | GradientBoostingClassifier | 0.992527          | 0.997055         | 0.943784       | 0.949737      |


### Model Score

The best boosted model is the **Light Gradient Boosted model**, which scored a **0.999435 test PR-AUC score**.

The PR-AUC evaluation metric highlights the model's:
- **High Precision**: The model achieves a high precision, meaning that when it predicts a transaction as fraudulent, it is very likely to be correct, reducing the number of false positives.
- **High Recall**: The model demonstrates high recall, meaning it correctly identifies a large proportion of actual fraudulent transactions, ensuring that as few fraudulent transactions as possible go undetected.

## Business Impact

- **Customer Experience**: With high precision and recall, customers are less likely to be inconvenienced by false alarms, and genuine fraud attempts are more likely to be caught early, protecting customers from potential financial harm.
- **Operational Efficiency**: Fewer false positives mean that fewer resources need to be spent on investigating transactions that aren’t fraudulent. This allows fraud analysts to focus on the transactions that are most likely to be fraudulent.
- **Risk Management**: High recall ensures that the model captures the majority of fraudulent transactions, which is crucial for minimizing financial losses and maintaining trust with customers.

## Contact

For any inquiries or questions, please contact:
- Evan Rosenbaum: [evanrosenbaum24@gmail.com]
