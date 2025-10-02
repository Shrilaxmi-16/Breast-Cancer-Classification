# Breast Cancer Classification with SVM

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white) 
![Pandas](https://img.shields.io/badge/Pandas-1.5-brightgreen) 
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-purple)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.2-orange)  


This repository contains a Python project demonstrates Support Vector Machine (SVM) classification using the Breast Cancer dataset. It includes preprocessing, training, visualization, hyperparameter tuning, and cross-validation.

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#HousingDataset)  
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Hyperparameter Tuning](#HyperparameterTuning)
- [Results](#Results) 
- [Tools & Libraries](#Tools&Libraries)
- [How to Run](#how-to-run)  
- [Conclusion](#conclusion)  
- [Author](#author)
   
## Project Overview

In this project, the dataset is first loaded and preprocessed through encoding and feature scaling to ensure uniform input for the model. The data is then divided into training and testing sets for evaluation. Support Vector Machines (SVM) are trained using both linear and RBF kernels to compare performance. To visualize the classifier's effectiveness, decision boundaries are plotted using a 2D PCA transformation of the feature space. Hyperparameters such as C and gamma are optimized using GridSearchCV to improve model performance. Finally, the model is evaluated using cross-validation, confusion matrix, and classification metrics including accuracy, precision, recall, and F1-score.


## 🏠 Housing Dataset 

**Dataset Source:** [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

The dataset used is the **Breast Cancer Dataset**.  

**Target (diagnosis):**  
- `M` = Malignant (1)  
- `B` = Benign (0)  

- id: Patient record identifier.
- diagnosis: Tumor diagnosis (M = Malignant, B = Benign).
- mean features: Average values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
- se features: Standard error values for the same set of features.
- worst features: Maximum (worst-case) values of the same features.
- Unnamed: 32: Empty column (drop as irrelevant).

## Exploratory Data Analysis

1. **Data Overview**
  - Loaded the dataset and examined basic statistics and structure.
  - Checked for missing values and inconsistencies to ensure data quality.

2. **Feature Distribution Analysis**
  - Visualized the distribution of numerical features using histograms and density plots.
  - Identified skewness and potential outliers in key features.

3. **Correlation Analysis**
  - Computed the correlation matrix for numerical features.
  - Visualized correlations using a heatmap to detect strong relationships between variables.

4. **Categorical Feature Analysis**
  - Created count plots for categorical variables to examine class distributions.
  - Observed imbalances and unique patterns in categorical features.

5. **Boxplots for Feature Comparison**
  - Compared numerical feature distributions against categorical variables.
  - Detected variations in feature values that may influence model predictions.

6. **Feature Importance Visualization**
  - Encoded categorical variables for model compatibility.
  - Applied a Random Forest Classifier to compute and rank feature importances.
  - Visualized importance scores using a bar chart with error bars for clarity.


## Hyperparameter Tuning

We use GridSearchCV with 5-fold cross-validation to find the best:

- C (Regularization parameter)
- gamma (Kernel coefficient for RBF)

## Results

- Linear SVM Accuracy: ~95% (varies with split)
- RBF SVM Accuracy: ~97–99% (after tuning)
- Best Hyperparameters: Selected using cross-validation
- The RBF kernel generally performs better than the linear kernel for this dataset.


## Tools & Libraries
- **Python**
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data visualization
- **Jupyter Notebook** – Interactive coding environment

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Shrilaxmi-16/Breast_Cancer_Analysis.git


## Conclusion

This project demonstrates how a Logestic Regression and Random Forest Classifier can be effectively used to perform feature importance analysis. Through exploratory data analysis, strong patterns and correlations between features were identified, and categorical variables were encoded for modeling. The Random Forest model ranked features based on their influence on predictions, providing valuable insights for feature selection and dimensionality reduction. Such analysis not only improves model performance but also enhances interpretability, enabling data-driven decision-making. This approach can be applied to a wide range of datasets to uncover the most impactful features and streamline predictive modeling workflows.

## Author
Shrilaxmi Gidd

Email: shrilaxmigidd16@gmail.com

Date: 02-10-2025
  
