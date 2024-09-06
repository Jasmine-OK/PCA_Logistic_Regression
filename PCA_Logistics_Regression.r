# Install required packages
if (!require("mlbench")) install.packages("mlbench")
if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("caret")) install.packages("caret")
if (!require("psych")) install.packages("psych")  # For descriptive stats

# Load required libraries
library(mlbench)
library(dplyr)
library(ggplot2)
library(caret)
library(psych)  # For descriptive statistics

# Step 1: Load the Breast Cancer dataset from mlbench
data(BreastCancer)
data <- BreastCancer

# Step 2: Data Exploration - Descriptive Statistics
cat("Descriptive Statistics of the Data:\n")
print(describe(data))  # Descriptive stats for initial insight

# Step 3: Data Preparation - Handle missing values
data <- data[complete.cases(data), ]

# Check for duplicates and remove them if any
cat("Checking for duplicates...\n")
data <- data %>% distinct()

# Step 4: Feature Engineering - Convert target variable to numeric
# Convert 'Class' to binary (2 = benign -> 0, 4 = malignant -> 1)
data$Class <- ifelse(data$Class == "benign", 0, 1)

# Remove ID column and convert other factor columns to numeric
data <- data %>%
  select(-Id) %>%
  mutate_if(is.factor, as.numeric)

# Step 5: Standardize the data
scaler <- preProcess(data[, -10], method = c("center", "scale"))
scaled_data <- predict(scaler, data[, -10])

# Step 6: PCA for dimensionality reduction (using 2 components)
pca <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

# Visualize the first two PCA components using a scatter plot
pca_data <- as.data.frame(pca$x[, 1:2])  # Select first 2 PCA components
pca_data$Class <- data$Class  # Add the class variable for color distinction

# Scatter plot of the PCA components
plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = as.factor(Class))) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA on Cancer Dataset",
       x = "First Principal Component",
       y = "Second Principal Component") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()

print(plot)  # Automatically display the plot

# Step 7 (Optional): Logistic Regression using PCA-reduced data
set.seed(123)
trainIndex <- createDataPartition(pca_data$Class, p = 0.8, list = FALSE)
train_data <- pca_data[trainIndex, ]
test_data <- pca_data[-trainIndex, ]

# Fit logistic regression model
logistic_model <- glm(Class ~ PC1 + PC2, family = binomial, data = train_data)

# Make predictions and evaluate performance
predictions <- predict(logistic_model, newdata = test_data, type = "response")
pred_class <- ifelse(predictions > 0.5, 1, 0)

# Calculate accuracy
accuracy <- mean(pred_class == test_data$Class)
cat("Logistic Regression Accuracy: ", round(accuracy * 100, 2), "%\n")

# Step 8: Interpretation of PCA and Results
cat("The first two principal components explain a significant portion of the variance in the dataset,\n")
cat("allowing us to reduce the dimensionality while retaining key information.\n")
cat("Logistic regression performed on the PCA-reduced data shows an accuracy of", round(accuracy * 100, 2), "%.\n")
cat("This indicates that PCA can be an effective dimensionality reduction technique for this dataset.\n")
cat("However, there is some overlap in the PCA space, suggesting that additional variables or\n")
cat("classification techniques might improve performance.\n")
