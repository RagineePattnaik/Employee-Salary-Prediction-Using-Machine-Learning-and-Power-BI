import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('employee_dataset.csv')
# Histogram ------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(df['Salary'], kde=True, color='teal', bins=30)
plt.title('Distribution of Salary', fontsize=16)
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

#Bar Chart--------------------------------------------------------------------------------------------------------------
# Initialize the LabelEncoder
encoder = LabelEncoder()
dataset = pd.read_csv('employee_dataset.csv')
# Encode the 'Country' column
dataset_encoded = dataset.copy()  # Ensure dataset_encoded is initialized
dataset_encoded['Country'] = encoder.fit_transform(dataset['Country'])

# Save the original mappings
country_mapping = dict(zip(encoder.transform(dataset['Country']), dataset['Country']))

# Add a decoded column for Country
dataset_encoded['Country_Names'] = dataset_encoded['Country'].map(country_mapping)

# Update age bins to exclude 0-20
age_bins = [20, 30, 40, 50, 60]  # Updated bin edges
age_labels = ['20-30', '30-40', '40-50', '50-60']  # Updated bin labels
dataset_encoded['Age_Binned'] = pd.cut(dataset_encoded['Age'], bins=age_bins, labels=age_labels, right=False)

# Ensure Experience Binned is already defined
exp_bins = [0, 3, 6, 9, 12, 15, 20]
exp_labels = ['0-3', '3-6', '6-9', '9-12', '12-15', '15-20']
dataset_encoded['Experience_Binned'] = pd.cut(dataset_encoded['Experience_Years'], bins=exp_bins, labels=exp_labels, right=False)

# Set the plot style
sns.set(style="whitegrid")

# Add histogram for Country vs Salary
plt.figure(figsize=(18, 6))

# Salary vs Age Groups
plt.subplot(1, 3, 1)
sns.barplot(data=dataset_encoded, x='Age_Binned', y='Salary', palette='coolwarm', ci=None)
plt.title('Average Salary vs Age Groups')
plt.xlabel('Age Groups')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)

# Salary vs Experience Groups
plt.subplot(1, 3, 2)
sns.barplot(data=dataset_encoded, x='Experience_Binned', y='Salary', palette='viridis', ci=None)
plt.title('Average Salary vs Experience Groups')
plt.xlabel('Experience Groups')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)

# Salary vs Country
plt.subplot(1, 3, 3)
sns.barplot(data=dataset_encoded, x='Country_Names', y='Salary', palette='magma', ci=None)
plt.title('Average Salary vs Country')
plt.xlabel('Country')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


#Box plot -------------------------------------------------------------------------------------------------------------
# List of numerical columns in your dataset (You can modify this list to include only relevant numerical columns)
numerical_columns = ['Salary', 'Age', 'Experience_Years', 'Performance_Score']

# Set the plot style
sns.set(style="whitegrid")

# Create box plots for each numerical column
plt.figure(figsize=(15, 12))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)  # Adjust the number of rows and columns depending on the number of variables
    sns.boxplot(data=dataset_encoded, x=column, palette='Set2')
    plt.title(f'Box Plot of {column}')
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()

# Pie Chart ---------------------------------------------------------------------------------------------------------
department_counts = df['Department'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Employee Distribution by Department', fontsize=16)
plt.show()


# Violin Chart-------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Education_Level', y='Salary', palette='coolwarm')
plt.title('Salary Distribution by Education Level', fontsize=16)
plt.xlabel('Education Level')
plt.ylabel('Salary')
plt.xticks(rotation=45)
plt.show()

# Multiline Chart-------------------------------------------------------------------------------------------------
avg_salary_by_dept_exp = df.groupby(['Experience_Years', 'Department'])['Salary'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_salary_by_dept_exp, x='Experience_Years', y='Salary', hue='Department', marker='o', palette='tab10')
plt.title('Average Salary by Experience Across Departments', fontsize=16)
plt.xlabel('Years of Experience')
plt.ylabel('Average Salary')
plt.legend(title='Department')
plt.show()

