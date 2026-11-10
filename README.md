# 🌸 Iris Flower Classification using Machine Learning

A machine learning project that predicts Iris flower species using three classification algorithms: Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN).

This project focuses on data preprocessing, exploratory data analysis, feature importance, model training, and performance evaluation.

## 📌 Project Overview

The Iris dataset contains measurements of three different Iris flower species.

Using these measurements, this project trains machine learning models to classify flowers into:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The performance of three classification algorithms is compared to identify the best-performing model.

## 🎯 Objectives

- Explore and understand the Iris dataset.
- Perform data preprocessing and label encoding.
- Analyze relationships between different features.
- Identify important features for classification.
- Train multiple machine learning models.
- Evaluate and compare model performance.
- Identify the best-performing classification algorithm.

## 🛠️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression
A supervised classification algorithm that predicts the probability of a data point belonging to a particular class.

### 2. Random Forest
An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reliability.

### 3. K-Nearest Neighbors (KNN)
A distance-based classification algorithm that predicts the class of a new data point based on its nearest neighbors.

## 🔍 Project Workflow

### 1. Data Loading & Preprocessing
- Load the Iris dataset using Pandas.
- Check dataset dimensions and missing values.
- Identify duplicate records.
- Remove unnecessary ID columns.
- Encode species labels using LabelEncoder.
- Split data into 80% training and 20% testing sets.

### 2. Exploratory Data Analysis (EDA)
- Analyze relationships between flower measurements.
- Generate a feature correlation matrix.
- Examine feature importance using Random Forest.
- Create pair plots to visualize relationships between features.

### 3. Model Training
Train three classification models:

- Logistic Regression
- Random Forest
- KNN (k=5)

StandardScaler is used with Logistic Regression and KNN to normalize feature values.

### 4. Model Evaluation

Evaluate the models using:

- Accuracy
- Precision
- Recall
- F1-Score
- 5-Fold Cross-Validation

### 5. Model Comparison
Compare the performance of all three algorithms and select the best model based on cross-validation accuracy, using test F1-score as a tie-breaker.

## 📊 Visualizations

### 1. Correlation Matrix

Shows the relationship between different flower features and their correlation with the target species.

![Correlation Matrix](01_correlation.png)

### 2. Feature Importance

Displays the importance of different features in the Random Forest model.

![Feature Importance](02_feature_importance.png)

### 3. Model Performance Comparison

Compares the Accuracy, Precision, Recall, and F1-score of all three models.

![Model Comparison](04_model_comparison.png)

### 4. Confusion Matrices

Displays the actual and predicted classifications for each model.

![Confusion Matrices](05_confusion_matrices.png)

## 📁 Project Structure

```text
MachineLearning/
│
├── iris_pipeline.py
├── model_comparison.csv
│
├── 01_correlation.png
├── 02_feature_importance.png
├── 04_model_comparison.png
├── 05_confusion_matrices.png
│
├── LICENSE
└── README.md
```

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/shreyashiijain/MachineLearning.git
```

### Step 2: Navigate to the Folder

```bash
cd MachineLearning
```

### Step 3: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Step 4: Add the Dataset

Download the Iris dataset and place `Iris.csv` inside the project folder.

### Step 5: Update File Paths

In `iris_pipeline.py`, update these lines:

```python
OUT = "./"
df = pd.read_csv("Iris.csv")
```

### Step 6: Run the Script

```bash
python iris_pipeline.py
```

The script will train the models, evaluate their performance, and generate visualizations and a model comparison CSV file.

## 📚 Key Learnings

- Understanding supervised machine learning.
- Data cleaning and preprocessing.
- Feature correlation and feature importance.
- Implementing classification algorithms.
- Training and testing machine learning models.
- Model evaluation using multiple performance metrics.
- Understanding cross-validation and confusion matrices.

## 👩‍💻 Author

**Shreyashi Jain**

- GitHub: [@shreyashiijain](https://github.com/shreyashiijain)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/shreyashi-jain-9a81a4333/)

---

⭐ If you find this project interesting, consider giving the repository a star!
