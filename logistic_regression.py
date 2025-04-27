import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Data Cleaning Function (modify as per your actual cleaning function)
def clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Check and handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    df = df.dropna()  # Drop rows with non-numeric missing values

    # Remove duplicates
    df = df.drop_duplicates()

    # Rename columns for clarity
    df.rename(
        columns={
            "Recency (months)": "Recency",
            "Frequency (times)": "Frequency",
            "Monetary (c.c. blood)": "Monetary",
            "Time (months)": "Time",
            "whether he/she donated blood in March 2007": "Donated_March_2007"
        },
        inplace=True,
    )

    # Remove outliers based on IQR for numeric columns
    numeric_cols = ['Recency', 'Frequency', 'Monetary', 'Time']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    
    return df

# Load and clean the data
file_path = 'transfusion.csv'
df = clean_data(file_path)

# Prepare features and target for classification
X = df[['Recency', 'Frequency', 'Time']]  # Features
y = df['Donated_March_2007']  # Target (binary classification)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression Classifier
log_reg_classifier = LogisticRegression(random_state=42)

# Train the classifier
log_reg_classifier.fit(X_train, y_train)

# Predictions and evaluation for Logistic Regression
y_pred_log_reg = log_reg_classifier.predict(X_test)

# Evaluate model performance
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Cross-validation for Logistic Regression
cv_scores_log_reg = cross_val_score(log_reg_classifier, X_scaled, y, cv=10, scoring='accuracy')
print("Logistic Regression Cross-validation Accuracy: ", cv_scores_log_reg.mean())  # Mean accuracy across folds

# Optional: Visualize the coefficients for feature importance
coefficients = log_reg_classifier.coef_[0]
features = X.columns
plt.bar(features, coefficients)
plt.title("Logistic Regression Feature Importance")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.show()
