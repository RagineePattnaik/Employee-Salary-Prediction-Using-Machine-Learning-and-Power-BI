# Association a d correlation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Set display options for better readability
desired_width = 400
pd.set_option('display.width', desired_width)  # sets the screen width to 400
pd.set_option('display.max_columns', 20)  # sets the max number of columns to display to 20

# Load the dataset
df = pd.read_csv(r'employee_dataset.csv')

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Display the correlation matrix
print(numeric_df.corr())

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Employee Dataset")
plt.show()

# Cluster Analysis --------------------------
import pandas as pd
from sklearn.cluster import KMeans

# Set display options for better readability
desired_width = 400
pd.set_option('display.width', desired_width)  # sets the screen width to 400
pd.set_option('display.max_columns', 20)  # sets the max number of columns to display to 20

# Load the dataset
df = pd.read_csv(r'employee_dataset.csv')

# Select only numeric columns for clustering (includes 'Salary' and any other numeric columns)
numeric_df = df.select_dtypes(include=['number'])

# Replace missing values with 0
numeric_df.fillna(0, inplace=True)

# Display the top three rows of the numeric DataFrame
print("Top 3 rows of numeric data for clustering:")
print(numeric_df.head(3))

# Apply KMeans clustering to create 5 groups
k_groups = KMeans(n_clusters=5, random_state=0).fit(numeric_df)

# Display cluster labels for each row
print("\nCluster labels for each row:")
print(k_groups.labels_)

# Display the number of rows and shape of the numeric DataFrame
print("\nNumber of rows in k_groups and shape of the DataFrame:")
print(len(k_groups.labels_), numeric_df.shape)

# Display the centroids for each cluster
print("\nCluster centroids (averages for each numeric column):")
print(k_groups.cluster_centers_)

# Display the centroid values for the first cluster
print("\nCentroid for the first cluster:")
print(k_groups.cluster_centers_[0])

# Add the cluster labels as a new column to the DataFrame
numeric_df['cluster'] = k_groups.labels_

# Display the top three rows of the data with cluster labels
print("\nTop 3 rows of the data with cluster labels:")
print(numeric_df.head(3))

# Display the mean of each numeric column grouped by clusters
print("\nMean values of each numeric column for each cluster:")
print(numeric_df.groupby('cluster').mean())


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv(r'employee_dataset.csv')

# Select only numeric columns (including 'Salary')
numeric_df = df.select_dtypes(include=['number'])

# Replace missing values with 0
numeric_df.fillna(0, inplace=True)

# Display the top three rows of the numeric DataFrame
print("Top 3 rows of numeric data for clustering:")
print(numeric_df.head(3))

# Loop to determine the best number of clusters (from 3 to 10)
for i in range(3, 10):
    k_groups = KMeans(n_clusters=i, random_state=0).fit(numeric_df)  # Fit KMeans with i clusters
    labels = k_groups.labels_
    sil_score = silhouette_score(numeric_df, labels)  # Calculate silhouette score
    print(f'K Groups = {i} | Silhouette Coefficient = {sil_score}')

# ----------------------------------------------------------- # End K Means clustering

#Feature engineering for Linear regression  ---------------------------

# 1. Interaction Feature: years_of_experience * performance_score
df['experience_performance_interaction'] = df['Experience_Years'] * df['Performance_Score']

# 2. Polynomial Feature: Square of years_of_experience (capturing non-linear effects)
df['experience_squared'] = df['Experience_Years'] ** 2

# 3. Polynomial Feature: Square of performance_score (capturing non-linear effects)
df['performance_squared'] = df['Performance_Score'] ** 2

# 4. Log Transformation (optional): Apply log transformation to salary if salary is highly skewed
df['log_salary'] = df['Salary'].apply(lambda x: np.log(x + 1))

# 5. Combine multiple features: years_of_experience + performance_score (capturing additive effects)
df['experience_plus_performance'] = df['Experience_Years'] + df['Performance_Score']

# View the modified dataframe with new features
print(df.head())

# Check correlation with salary for the newly created features
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
print(correlation_matrix['Salary'].sort_values(ascending=False))

# Visualize correlations using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix with Interaction Features")
plt.show()

# ----------------------------------------------------------- # End of Feature Engineering

#Linear regression --------------------------------
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Feature engineering - assuming 'log_salary' is already created
X = df[['log_salary']]  # Independent variable (log_salary)
y = df['Salary']        # Target variable (Salary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
linear_reg = LinearRegression()

# Train the Linear Regression model
linear_reg.fit(X_train, y_train)

# Predict target values for the test set
y_pred = linear_reg.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
n = len(y_test)
p = X_train.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display metrics
print("Linear Regression Metrics:")
print(f"Adjusted R²: {adjusted_r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Visualize Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title("Actual vs Predicted Salary")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.grid(True)
plt.show()

#Logistic regression--------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Data Preprocessing
# 1. Creating a classification target: e.g., predict high vs. low salary
df['salary_class'] = df['Salary'].apply(lambda x: 'low' if x < 80000 else 'high')
print(df.salary_class.value_counts())

# 2. Selecting relevant features
X = df[['Experience_Years', 'Performance_Score', 'Age','log_salary']]  # Include numerical predictors
y = df['salary_class']  # Target variable
y2 = df.salary_class
log = LogisticRegression()
print(log.fit(X,y))
print(log.score(X,y2))
X_train, X_test, y2_train, y2_test = train_test_split(X, y2)
print(X_train.shape, y2_train.shape, X_test.shape, y2_test.shape)
print(log.fit(X_train, y2_train))
print(log.score(X_test, y2_test))

#3. Confusion matrix

log = LogisticRegression()
X_train, X_test, y2_train, y2_test = train_test_split(X, y2)

log.fit(X_train, y2_train)
log.score(X_test, y2_test)
y2_pred = log.predict(X_test)
print(y2_pred, np.array(y2_test))
print(confusion_matrix(y2_test, y2_pred))

# ----------------------------------------------------------- # End of Logistic Regression

# KNN -----------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Define features (X) and target variable (y)
X = df[['log_salary', 'Experience_Years', 'Performance_Score']]  # Features, adjust according to your dataset
y = df['Salary']  # Target variable

# Create a classification target variable for salary: 'low' if Salary < 75000, else 'high'
df['salary_class'] = df['Salary'].apply(lambda x: 'low' if x < 75000 else 'high')
y2 = df['salary_class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # For regression
X1_train, X1_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)  # For classification

# Initialize KNN models
knn_reg = KNeighborsRegressor()  # KNN for regression
knn_class = KNeighborsClassifier()  # KNN for classification

# Train the KNN regression model
knn_reg.fit(X_train, y_train)  # Fitting the regression model with training data
print('KNN Regression score = ', knn_reg.score(X_test, y_test))  # Displaying R2 score for regression

# Train the KNN classification model
knn_class.fit(X1_train, y2_train)  # Fitting the classification model with training data
print('KNN Classification score = ', knn_class.score(X1_test, y2_test))  # Displaying accuracy score for classification

# Predict with KNN classification model
y2_pred = knn_class.predict(X1_test)

# Display confusion matrix for classification model
print("\nConfusion Matrix for KNN Classification:")
print(confusion_matrix(y2_test, y2_pred))  # True positives and false positives on the diagonal


# ----------------------------------------------------------- # End of KNN

# PCA-----------------------------------------------------------
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define features (X) and target variable (y)
X = df[['log_salary', 'Experience_Years', 'Performance_Score']]  # Adjust columns based on your dataset
y = df['Salary']  # Target variable

# Apply PCA to reduce dimensions
pca = PCA(n_components=3)  # Keep 3 principal components (you can adjust this as needed)
X_transformed = pca.fit_transform(X)

# Split the dataset into training and testing sets for PCA-transformed data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Print the shapes of transformed and split datasets
print(X_transformed.shape)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # Displaying the sizes of train and test sets

# Initialize the Linear Regression model
lg = LinearRegression()

# Train the Linear Regression model with PCA-transformed data
lg.fit(X_train, y_train)

# Evaluate the model's performance on the test data (PCA)
print('PCA Linear Regression R-squared = ', lg.score(X_test, y_test))  # Display R^2 score on the test set

# Now, evaluate the model using non-PCA data for comparison
X_train_non_pca, X_test_non_pca, y_train_non_pca, y_test_non_pca = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model with non-PCA data
lg.fit(X_train_non_pca, y_train_non_pca)

# Evaluate the model's performance on the test data (non-PCA)
print('Non-PCA Linear Regression R-squared = ', lg.score(X_test_non_pca, y_test_non_pca))  # Display R^2 score on the test set



#NLP --------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a synthetic text column combining Country, Department, and Position
df['Employee_Info'] = df['Country'] + ' ' + df['Department'] + ' ' + df['Position']

# Step (0): Assigning a variable to the CountVectorizer() function
count_vect = CountVectorizer(ngram_range=(1, 3), max_features=1000)  # Unigrams, bigrams, and trigrams
X_count = count_vect.fit_transform(df['Employee_Info'])  # Transforms the Employee_Info into unique word count vectorization

# Step (1): Displays the type and shape of X
print(f'CountVectorizer Shape: {X_count.shape}')
print(f'CountVectorizer Type: {type(X_count)}')

# Step (2): Assigns the 'salary_class' column as y (target variable)
y = df['salary_class']

# Step (3): Randomly splits X and y into model X/y training and test subsets
X_train_count, X_test_count, y_train, y_test = train_test_split(X_count, y, test_size=0.2, random_state=42)

# Step (4): Assigning a variable to the TFIDFVectorizer() function
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)  # Unigrams, bigrams, and trigrams
X_tfidf = tfidf.fit_transform(df['Employee_Info'])  # Transforms into unique word TFIDF vectorization

# Step (5): Transforms into unique word TFIDF vectorization using unigrams, bigrams, and trigrams
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step (6): Displays the type and shape of X1 (TFIDF transformed)
print(f'TFIDF Vectorizer Shape: {X_tfidf.shape}')
print(f'TFIDF Vectorizer Type: {type(X_tfidf)}')

# Step (7): Train a Logistic Regression model using CountVectorizer (X_count)
log_reg_count = LogisticRegression(max_iter=1000)
log_reg_count.fit(X_train_count, y_train)
y_pred_count = log_reg_count.predict(X_test_count)

# Step (8): Train a Logistic Regression model using TFIDF Vectorizer (X_tfidf)
log_reg_tfidf = LogisticRegression(max_iter=1000)
log_reg_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = log_reg_tfidf.predict(X_test_tfidf)

# Step (9): Displaying R2 score for both models
print(f'Count Vectorizer Model Accuracy: {accuracy_score(y_test, y_pred_count)}')
print('Classification Report (Count Vectorization):')
print(classification_report(y_test, y_pred_count))

print(f'TFIDF Model Accuracy: {accuracy_score(y_test, y_pred_tfidf)}')
print('Classification Report (TFIDF):')
print(classification_report(y_test, y_pred_tfidf))

# Step (10): Calculate confusion matrices for each model
print('Confusion Matrix (Count Vectorization):')
print(confusion_matrix(y_test, y_pred_count))

print('Confusion Matrix (TFIDF):')
print(confusion_matrix(y_test, y_pred_tfidf))

