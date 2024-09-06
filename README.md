# README: Principal Component Analysis (PCA) and Logistic Regression on Cancer Dataset

## Overview
This project demonstrates how Principal Component Analysis (PCA) can be used to extract essential features from the breast cancer dataset and reduce its dimensionality. Logistic regression is then optionally applied to classify whether a tumor is malignant or benign. Both Python and R implementations are provided to achieve these tasks. 

The breast cancer dataset contains various features describing cancer cells, and the target variable indicates whether the tumor is malignant (1) or benign (0).

## Data Source
The breast cancer dataset is sourced from `sklearn.datasets` in Python and from the `MASS` package in R. This dataset contains real-world data on breast cancer diagnoses, making it suitable for demonstrating dimensionality reduction techniques and classification models.

## Tasks
1. **PCA Implementation**: Perform Principal Component Analysis (PCA) on the breast cancer dataset to extract essential features.
2. **Dimensionality Reduction**: Reduce the dataset to 2 principal components for better visualization and understanding.
3. **Logistic Regression**: Apply logistic regression to classify whether the tumor is malignant or benign based on the PCA-reduced dataset.

## Available Files

1. **PCA_Logistics_Regression.py**: Python implementation of PCA and Logistic Regression.
2. **PCA_Logistics_Regression.r**: R implementation of PCA and Logistic Regression.
3. **Figure_1.png**: PCA visualization plot of the two principal components, color-coded to show whether tumors are malignant or benign.

## Detailed Steps in Python and R

### 1. **Loading the Dataset**:
   - **Python**: The breast cancer dataset is loaded using `load_breast_cancer()` from `sklearn.datasets`. It contains 30 features such as radius, texture, and smoothness, which describe the physical properties of the tumors.
   - **R**: The same dataset is accessed via the `MASS` package, where it is stored in a similar structure, with 30 features and a binary target variable (malignant or benign).

### 2. **Descriptive Statistics**:
   - **Python**: Descriptive statistics are generated using `X.describe()`. This includes the count, mean, standard deviation, and min/max values for each feature. This step helps identify if any features have extreme values or outliers that could affect the PCA.
   - **R**: The `summary()` function is used to generate descriptive statistics, which serve the same purpose as in Python. It provides insight into the range and distribution of each feature.
   
   **Interpretation**: Understanding the data distribution is critical, especially before applying PCA. For instance, if a feature has a much larger range than others, it could dominate the PCA unless standardized, which brings us to the next step.

### 3. **Checking for Duplicates**:
   - **Python**: The dataset is checked for duplicate rows using `X.duplicated().sum()`. If duplicates exist, they are removed to ensure that the analysis is not biased by repeated data points.
   - **R**: The dataset is also checked for duplicates using R’s functions, and any found are dropped.
   
   **Interpretation**: Removing duplicates is important because they can skew the results by giving too much weight to certain data points, especially in PCA, where each data point influences the computation of components.

### 4. **Standardization**:
   - **Python**: The dataset is standardized using `StandardScaler()` from `sklearn.preprocessing`. Standardization is essential for PCA because it ensures that each feature has a mean of 0 and a standard deviation of 1, preventing features with larger ranges from dominating the principal components.
   - **R**: In R, standardization is performed using `preProcess()` from the `caret` package. This function scales all features to ensure they contribute equally to the PCA.
   
   **Interpretation**: Standardizing the data is a crucial transformation step before applying PCA. Without it, features with large values (e.g., radius) could disproportionately influence the principal components, making it difficult to capture the true structure of the data.

### 5. **Principal Component Analysis (PCA)**:
   - **Python**: PCA is performed using `PCA()` from `sklearn.decomposition`. The dataset is reduced to two principal components, which together capture a significant portion of the variance in the dataset. The exact percentage of variance captured by each component is printed using `explained_variance_ratio_`.
   - **R**: In R, PCA is performed using the `prcomp()` function, which similarly reduces the dataset to two components. The amount of variance captured by these components is printed to help assess the effectiveness of the dimensionality reduction.
   
   **Interpretation**: In both implementations, the two principal components should capture around 63-65% of the dataset’s variance. This means that while we lose some information by reducing the dimensions, the most important patterns in the data are retained.

### 6. **Visualization**:
   - **Python**: The two principal components are visualized using `matplotlib`. A scatter plot is generated where each data point represents a tumor, and the points are color-coded based on whether the tumor is malignant or benign.
   - **R**: The same scatter plot is generated using `ggplot2`, with tumors color-coded by their classification.
   
   **Interpretation**: The PCA scatter plot provides a visual representation of how well the PCA separates malignant and benign tumors. If the points are well-separated into clusters, it suggests that PCA has effectively captured the variance in the dataset. The color-coding helps us visually assess the performance of PCA in distinguishing between the two classes.

### 7. **Logistic Regression **:
   - **Python**: Logistic regression is applied to the PCA-reduced dataset using `LogisticRegression()` from `sklearn.linear_model`. The model is trained on 80% of the data and tested on the remaining 20%. The accuracy of the model is calculated using `accuracy_score()`.
   - **R**: Logistic regression is performed using `glm()` from the `e1071` package in R. The dataset is split into training and testing sets using `sample.split()` from the `caTools` package, and accuracy is printed after model evaluation.
   
   **Interpretation**: Logistic regression is used to classify tumors as malignant or benign based on the two PCA components. The accuracy score provides a measure of how well the model performs. A high accuracy indicates that the PCA components contain enough information to classify the data effectively, despite the reduction in dimensionality.

### 8. **Accuracy Evaluation**:
   - **Python**: The logistic regression model’s accuracy is printed, showing the percentage of correct classifications on the test set.
   - **R**: Similarly, the accuracy of the logistic regression model is calculated and printed in R.

   **Interpretation**: The accuracy score helps determine how well the logistic regression model performs with the reduced dataset. While PCA simplifies the data by reducing its dimensions, a high accuracy indicates that the most important patterns in the data were retained.

### 9. **Visualization File (Figure_1.png)**:
   - This image contains the scatter plot generated from the PCA, showing the separation between malignant and benign tumors in the two principal components.

### How to Run the Code

1. **Python**:
   - Ensure you have Python installed along with the necessary libraries (`pandas`, `numpy`, `matplotlib`, `scikit-learn`).
   - Download and navigate to the directory containing `PCA_Logistics_Regression.py`.
   - Run the script using a Python interpreter:
     ```bash
     python PCA_Logistics_Regression.py
     ```

2. **R**:
   - Ensure you have R installed along with the necessary packages (`caret`, `ggplot2`, `e1071`, `MASS`, `caTools`).
   - Download and navigate to the directory containing `PCA_Logistics_Regression.r`.
   - Open RStudio or any R IDE, load the script, and run it.

### Conclusion

This project demonstrates how PCA can effectively reduce the dimensionality of a dataset while retaining the most important features. By applying logistic regression to the reduced dataset, we achieve accurate classification of tumors as malignant or benign. The project is implemented in both Python and R, and the results are visualized using `Figure_1.png` to understand the effectiveness of PCA.

