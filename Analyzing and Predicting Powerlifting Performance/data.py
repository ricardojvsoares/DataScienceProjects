import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set style for seaborn
sns.set(style="whitegrid")

# File paths
openpowerlifting_path = "Analyzing and Predicting Powerlifting Performance\\openpowerlifting.csv"
meets_path = "Analyzing and Predicting Powerlifting Performance\\meets.csv"

# Load datasets
powerlifting_df = pd.read_csv(openpowerlifting_path)
meets_df = pd.read_csv(meets_path)

# Print the column names to verify
print("\nPowerlifting Dataset Columns:")
print(powerlifting_df.columns)

# Overview of the datasets
print("\nPowerlifting Dataset Overview:")
print(powerlifting_df.head())
print("\nPowerlifting Dataset Info:")
print(powerlifting_df.info())

# Check for missing values
print("\nMissing Values in Powerlifting Data:")
print(powerlifting_df.isnull().sum())

print("\nMissing Values in Meets Data:")
print(meets_df.isnull().sum())

# Handle missing values (example: dropping rows with missing values)
powerlifting_df.dropna(inplace=True)
meets_df.dropna(inplace=True)

# Convert relevant columns to appropriate data types (if necessary)
# For example: Convert 'date' column in meets_df to datetime if it exists
if 'date' in meets_df.columns:
    meets_df['date'] = pd.to_datetime(meets_df['date'])

# Exploratory Data Analysis (EDA)

# Statistical summary of the powerlifting data
print("\nStatistical Summary of Powerlifting Data:")
print(powerlifting_df.describe())

# Define the correct total lift column name
total_column_name = 'TotalKg'  # Using the correct column name

# Distribution of total lifts
plt.figure(figsize=(12, 6))
sns.histplot(powerlifting_df[total_column_name], bins=30, kde=True)
plt.title('Distribution of Total Lifts')
plt.xlabel('Total Lift (kg)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Box plot of total lifts by sex
plt.figure(figsize=(12, 6))
sns.boxplot(x='Sex', y=total_column_name, data=powerlifting_df)
plt.title('Total Lifts by Sex')
plt.xlabel('Sex')
plt.ylabel('Total Lift (kg)')
plt.grid()
plt.show()

# Trend of total lifts over the years (if applicable)
if 'Year' in meets_df.columns:
    meets_df['Year'] = meets_df['Year'].dt.year
    average_lifts_per_year = powerlifting_df.groupby(meets_df['Year']).mean()[total_column_name].reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=average_lifts_per_year, x='Year', y=total_column_name)
    plt.title('Average Total Lifts Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Average Total Lift (kg)')
    plt.grid()
    plt.show()

# Prepare data for modeling (using relevant features)
features = ['WeightClassKg', 'Age', 'Sex']  # Include more features as necessary
X = pd.get_dummies(powerlifting_df[features], drop_first=True)  # Convert categorical variables
y = powerlifting_df[total_column_name]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plotting actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.title('Actual vs Predicted Total Lifts')
plt.xlabel('Actual Total Lift (kg)')
plt.ylabel('Predicted Total Lift (kg)')
plt.grid()
plt.show()
