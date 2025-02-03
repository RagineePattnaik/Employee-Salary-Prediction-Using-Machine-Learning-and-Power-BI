import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

df = pd.read_csv(r'employee_dataset.csv')  # reads Zillow file
print(df.describe())

# lists columns, row counts, and data types
print(df.info())
# Check for null values
null_values = df.isnull().sum()

# Display the count of null values in each column
print("Null Values in each column:")
print(null_values)

# Example of columns to check for outliers
columns_to_check = ['Salary', 'Performance_Score', 'Experience_Years','Age']

# Boxplot for each column to visualize outliers
for column in columns_to_check:
    sn.boxplot(df[column])
    plt.title(f"Boxplot for {column}")
    plt.show()

# Using IQR method to identify outliers for each of the chosen columns
for column in columns_to_check:
    Q1 = df[column].quantile(0.25)  # First quartile
    Q3 = df[column].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile range

    # Define outliers as values outside of [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"Outliers in {column}:")
    print(outliers)


# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Check the correlation between all numerical columns
correlation_matrix = numeric_df.corr()

# Print correlation of 'Salary' with all other columns
print("Correlation of 'Salary' with other numeric columns (sorted):")
print(correlation_matrix.loc['Salary', :].sort_values(ascending=False))

# Alternatively, check the correlation of 'Experience_Years' with other columns
print("\nCorrelation of 'Experience_Years' with other numeric columns (sorted):")
print(correlation_matrix.loc['Experience_Years', :].sort_values(ascending=False))


# OLAP slicing
# Display unique departments to identify a slicing dimension
print(df['Department'].unique())

# Slice the dataset based on Salary and other dimensions
# Example: Select employees with Salary <= 50000, Age > 30, and in the HR department
filtered_df = df[(df['Salary'] <= 50000) & (df['Age'] > 30) & (df['Department'] == 'HR')]

# Display the result
print(filtered_df)





