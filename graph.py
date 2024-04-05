# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt  # for visualizations

# Load data (replace 'ecommerce_data.csv' with your actual file path)
data = pd.read_csv('ecommerce-data.csv')

# Data Cleaning

# Check for missing values
print("Missing values summary:")
print(data.isnull().sum())

# Handle missing values (replace with your logic based on data)
# For example, if 'Total_Orders' has missing values, you might decide to impute them with the mean
# data['Total_Orders'].fillna(data['Total_Orders'].mean(), inplace=True)

# Check for data inconsistencies (e.g., negative values where unexpected)
# ... (add checks based on your data)

# Exploratory Data Analysis (EDA)
# Analyze customer demographics (assuming columns exist)
data.describe(include='all').transpose()  # summarize numerical and categorical data

# Analyze purchase history (assuming columns exist)
plt.hist(data['Total_Orders'])  # distribution of total orders
plt.xlabel('Total Orders')
plt.ylabel('Number of Customers')
plt.title('Distribution of Total Orders')
plt.show()

# Visualize churn rate by customer segments (e.g., location)
plt.pie(data['Churn'].value_counts(), labels=['Not Churned', 'Churned'], autopct='%1.1f%%')
plt.title('Churn Rate')
plt.show()

# Feature Engineering

# Create new features based on existing data
data['Avg_Order_Value'] = data['Total_Amount'] / data['Total_Orders']  # Assuming 'Total_Amount' exists
data['Customer_Age_Group'] = pd.cut(data['Age'], bins=[18, 25, 35, 45, 65], labels=['18-24', '25-34', '35-44', '45-64'])
# Feature Selection (consider correlation analysis or feature importance techniques)
# For now, select features based on domain knowledge and EDA insights
features = ['Total_Orders', 'Avg_Order_Value', 'Customer_Age_Group']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['Churn'], test_size=0.2, random_state=42)

# Build the model (Logistic Regression in this example)
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Analyze model coefficients to understand feature importance (for Logistic Regression)
coefficients = pd.DataFrame(model.coef_.T, data.columns[features])
coefficients.columns = ['Coefficient']
print("Feature Coefficients:")
print(coefficients)
