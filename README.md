# Machine Learning Assignment 2 – Classification Models

## 1. Problem Statement

The objective of this project is to implement and compare five machine learning classification algorithms using the Dry Bean dataset.

The project evaluates the performance of different classification techniques in predicting the variety of dry beans based on their physical characteristics.

The five classification algorithms implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset Name:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

Dataset characteristics:

- Total instances: 13,611
- Number of input features: 16
- Target variable: Class
- Number of classification categories: 7
- Problem type: Multi-class classification

The dataset contains numerical measurements describing the physical characteristics of dry beans. These measurements are used to predict the corresponding bean variety.

## 3. GitHub Repository Link

Repository URL: [ADD YOUR GITHUB REPOSITORY LINK]

The repository contains:

- app.py – Streamlit application
- requirements.txt – Python dependencies
- test_data.csv – Test dataset
- model_results.csv – Model evaluation results
- models/ – Five saved classification models
- README.md – Project documentation

## 4. Models Used and Evaluation

The dataset was divided into 80% training data and 20% testing data using stratified sampling.

Random state: 42

Feature scaling was applied to Logistic Regression, KNN and SVM using StandardScaler.

The following table presents the evaluation results of all five classification models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Observations on Model Performance

**Logistic Regression**

Logistic Regression achieved 92.14% accuracy and an AUC score of 0.9934. It demonstrated strong classification performance across the evaluation metrics.

**Decision Tree**

Decision Tree achieved 89.17% accuracy and an AUC score of 0.9330. It performed reasonably well but showed lower overall performance compared to Logistic Regression, KNN and SVM.

**K-Nearest Neighbors**

KNN achieved 91.66% accuracy and an AUC score of 0.9812. It provided good classification performance by utilizing the similarity between neighboring data points.

**Naive Bayes**

Naive Bayes achieved 76.39% accuracy. It recorded the lowest accuracy among the five models. The conditional independence assumption of Naive Bayes may limit its performance when input features are correlated.

**Support Vector Machine**

SVM achieved the highest accuracy of 92.21% and the highest AUC score of 0.9937. It also achieved the best Precision, Recall, F1 Score and MCC among the five models.

### Overall Winner for the Dataset

**Support Vector Machine (SVM)**

Based on the experimental results, Support Vector Machine is the best-performing classifier for this dataset.

It achieved the highest Accuracy (92.21%), AUC (0.9937), Precision (92.24%), Recall (92.21%), F1 Score (92.22%) and MCC (0.9059).

However, Logistic Regression also demonstrated comparable performance, with only a small difference in accuracy.

## 5. Streamlit Application

The project includes an interactive Streamlit application with the following features:

1. Upload test dataset in CSV format.
2. Select one of the five trained classification models.
3. Display Accuracy, AUC, Precision, Recall, F1 Score and MCC.
4. Display the confusion matrix.
5. Display the classification report.
6. View actual and predicted classification results.
7. Download prediction results as a CSV file.

The application loads previously trained models and evaluates the uploaded test dataset without retraining.

**Live Streamlit Application:** [ADD YOUR DEPLOYED STREAMLIT LINK]

## 6. Technologies Used

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

## 7. Conclusion

This project implemented and evaluated five different machine learning classification algorithms using the Dry Bean dataset.

The experimental results indicate that Support Vector Machine achieved the best overall performance, followed closely by Logistic Regression and KNN.

Naive Bayes recorded the lowest classification accuracy among the five algorithms.

The project demonstrates the implementation, evaluation, comparison and deployment of machine learning classification models through an interactive Streamlit web application.
