# XLGBM-Fraud-Detect

## Overview

My submission uses a stacked ensemble model combining <strong>XGBoost</strong> and <strong>LightGBM</strong> classifiers to enhance predictive performance. The stacking approach leverages multiple base models to improve overall accuracy by combining their strengths.

## Data Preprocessing

In this data preprocessing pipeline, several steps are undertaken to clean and prepare the dataset for machine learning:

<strong>Dropping Irrelevant Columns</strong>:

Columns such as step, customer, zipcodeOri, and zipMerchant are dropped from the dataset as they may not provide meaningful insights for the model.
Handling Missing and Categorical Data:

The age column, which contains the value 'U' for unknown ages, is cleaned by replacing 'U' with NaN, removing any quotation marks, and converting the data to a numerical format. Missing values in age are then filled with the median age.

Categorical columns like gender, merchant, and category are encoded using LabelEncoder to convert text labels into numerical values that can be used by machine learning models.

<strong>Scaling Numerical Data</strong>:

The amount column is standardized using StandardScaler to ensure that all features contribute equally to the model and to improve convergence during training.

<strong>Dealing with Class Imbalance</strong>

In this dataset, only 1.2% of the transactions are fraudulent, while 98.8% are not. Such a severe class imbalance can lead to misleading results if accuracy is used as the primary evaluation metric. For instance, a model that predicts all transactions as non-fraudulent would achieve 98.8% accuracy but would be completely ineffective at identifying fraud.

To address this, <strong>Synthetic Minority Over-sampling Technique (SMOTE)</strong> was used. SMOTE artificially increases the number of fraudulent transactions by generating synthetic samples, balancing the dataset and improving the model's ability to detect fraud. This technique helps ensure that the model learns from both classes effectively, making metrics like precision, recall, and F1 score more reliable indicators of performance in this context.

## Models used

##### 1. XGBoost Classifier (xgb_model)

Type|XGBClassifier
<br/>Parameters:

- colsample_bytree: 1.0
- learning_rate: 0.3
- max_depth: 10
- n_estimators: 600
- subsample: 1.0
- gamma: 2.0
- min_child_weight: 1
- tree_method: 'gpu_hist'
- use_label_encoder: False

This model uses gradient boosting with GPU acceleration to handle large datasets and complex patterns efficiently. The chosen parameters aim to balance model complexity and training stability.

##### 2. LightGBM Classifier (lgbm_model)

Type|LGBMClassifier
<br/>Parameters:

- n_estimators: 500
- learning_rate: 0.01
- max_depth: 10

LightGBM is used for its efficiency and speed in handling large datasets. The learning rate is set lower to ensure gradual model training and avoid overfitting.

##### 3. Stacking Classifier (stacking_clf)

Type|StackingClassifier
<br/><strong>Base Models</strong>:

1. xgb_model: XGBoost classifier
2. lgbm_model: LightGBM classifier

<strong>Final Estimator</strong>:
A second XGBoost model with the same parameters as xgb_model
The stacking classifier combines the predictions from xgb_model and lgbm_model, using them as input to a final XGBoost model that refines the predictions for improved accuracy.

![Stacking Classifier](https://raw.githubusercontent.com/siddheshtv/hsbc-hackathon/main/image.png)

## Evaluation Metrics

<strong>Type: <em>Train</em></strong>

| Metric                            | Score  |
| --------------------------------- | ------ |
| Precision                         | 0.9796 |
| Recall                            | 0.9915 |
| F1 Score                          | 0.9855 |
| AUC-ROC                           | 0.9987 |
| True Negatives (TN)               | 99293  |
| False Positives (FP)              | 2100   |
| False Negatives (FN)              | 862    |
| True Positives (TP)               | 100723 |
| True Positive Rate (TPR) / Recall | 0.9915 |
| False Positive Rate (FPR)         | 0.0207 |

## Analysis Overview

![Transaction Amount Distribution](https://raw.githubusercontent.com/siddheshtv/hsbc-hackathon/main/image-1.png)
![Transaction Amount by Merchant Category](https://raw.githubusercontent.com/siddheshtv/hsbc-hackathon/main/image-2.png)
![Fraud vs Not Fraud](https://raw.githubusercontent.com/siddheshtv/hsbc-hackathon/main/image-3.png)
![Merchant ID Distribution](https://raw.githubusercontent.com/siddheshtv/hsbc-hackathon/main/image-4.png)
