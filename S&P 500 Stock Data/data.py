# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "DataScienceProjects\S&P 500 Stock Data\\all_stocks_5yr.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Check for null values
print("\nMissing Values:")
print(data.isnull().sum())

# Get dataset summary
print("\nDataset Summary:")
print(data.describe())

# Check the data types
print("\nData Types:")
print(data.dtypes)

# Convert 'date' to datetime format
data['date'] = pd.to_datetime(data['date'])

# Sort the data by 'date'
data = data.sort_values(by='date')

# Display unique stock names
print("\nUnique Stock Names:")
print(data['Name'].unique())

# Visualize the volume of trades for the first stock
stock_name = data['Name'].iloc[0]
stock_data = data[data['Name'] == stock_name]

plt.figure(figsize=(12, 6))
plt.plot(stock_data['date'], stock_data['volume'], label=f"Volume of {stock_name}")
plt.title(f"Trading Volume of {stock_name} Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()

# Correlation heatmap
print("\nCorrelation Heatmap:")
numeric_columns = ['open', 'high', 'low', 'close', 'volume']
correlation = data[numeric_columns].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Between Numeric Features")
plt.show()

# Convert 'date' to datetime and sort by date
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

# Feature Engineering: Add new features
data['price_change'] = data['close'] - data['open']  # Price change
data['daily_return'] = data['close'].pct_change()    # Daily return
data['moving_avg_5'] = data['close'].rolling(window=5).mean()  # 5-day moving average
data['volatility'] = data['high'] - data['low']     # Intraday volatility

# Drop rows with NaN values created by rolling or pct_change
data = data.dropna()

# Target Variable: 1 if price increased, 0 otherwise
data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

# Drop unnecessary columns
features = ['open', 'high', 'low', 'close', 'volume', 'price_change', 
            'daily_return', 'moving_avg_5', 'volatility']
X = data[features]
y = data['target']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()