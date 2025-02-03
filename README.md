# Employee-Salary-Prediction-Using-Machine-Learning-and-Power-BI
# Employee Salary Prediction

## Project Overview

This project leverages data analysis and machine learning techniques to predict employee salaries based on various factors, including demographics, job roles, education, and performance metrics. The primary goal is to provide organizations with data-driven insights to optimize compensation strategies.

## Key Features

1. **Data Preparation**:
   - Extensive preprocessing to ensure no missing values or outliers.
   - Feature engineering with interaction terms, polynomial features, and log transformations to enhance predictive power.

2. **Data Exploration**:
   - Conducted exploratory data analysis (EDA) to uncover hidden patterns and relationships using visualizations such as histograms, bar charts, box plots, and violin plots.

3. **Predictive Modeling**:
   - Utilized multiple machine learning models:
     - **Linear Regression**: Achieved an R² of 0.98, making it the most effective model.
     - **Logistic Regression**: Classified salaries into "Low" and "High" categories.
     - **K-Nearest Neighbors (KNN)**: Provided a high classification accuracy of 98.4%.
     - **Natural Language Processing (NLP)**: Extracted features from categorical data using CountVectorizer and TF-IDF.

4. **Clustering**:
   - Applied K-Means clustering to identify patterns in salary, performance, and experience, resulting in an optimal cluster count of three.

5. **Visualization**:
   - Developed an interactive Power BI dashboard showcasing workforce insights, salary trends, and experience distribution.

## Dataset Details

The dataset includes the following columns:
- **Employee_ID**: Unique identifier for each employee.
- **First_Name** & **Last_Name**: Employee's name.
- **Department**: Department of employment (e.g., HR, IT, Finance).
- **Position**: Job role (e.g., Manager, Engineer).
- **Age**: Employee's age.
- **Salary**: Target variable for prediction.
- **Joining_Date**, **City**, **Country**: Additional demographic data.
- **Performance_Score**: Numeric performance rating.
- **Experience_Years**: Total years of professional experience.
- **Education_Level**: Highest education attained.
- **Gender** & **Marital_Status**: For diversity analysis.

## Power BI Dashboard Highlights

- **Workforce Overview**:
  - 1,009 employees with an average age of 41 years, average salary of $79,000, and 17 years of experience.
  - Gender distribution: 664 males and 682 females.
- **Department Insights**:
  - Operations department accounts for the highest total salaries ($33.8M), followed by Marketing and HR.
- **Interactivity**:
  - Filters based on education levels and other features for tailored analysis.

## Models and Evaluation

1. **Linear Regression**:
   - R²: 0.98
   - RMSE: 3,263.27
   - Best-performing model for precise salary predictions.

2. **Logistic Regression**:
   - Classified salaries into "Low" (< $80,000) and "High" (≥ $80,000).
   - High accuracy but limited insights compared to regression.

3. **KNN Classification**:
   - Accuracy: 98.4%
   - Balanced performance with slight overfitting, requiring hyperparameter tuning.

4. **Natural Language Processing (NLP)**:
   - CountVectorizer and TF-IDF vectorization techniques applied.
   - Accuracy: ~55%, limited effectiveness due to non-textual nature of data.

## Future Scope

- **Attrition Analysis**: Incorporate attrition data to predict turnover patterns.
- **Feature Expansion**: Add behavioral, job satisfaction, and psychological factors for improved accuracy.
- **Model Refinement**: Optimize existing models through advanced hyperparameter tuning.

## Repository Structure

```plaintext
.
├── Data/
│   ├── raw_dataset.csv
│   └── processed_dataset.csv
├── Models/
│   ├── linear_regression_model.pkl
│   ├── logistic_regression_model.pkl
│   └── knn_model.pkl
├── Notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
├── PowerBI_Dashboard/
│   └── Employee_Salary_Dashboard.pbix
├── README.md
└── requirements.txt              

