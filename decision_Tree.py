import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
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

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Predictions and evaluation for Decision Tree
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate model performance
print("Decision Tree Accuracy: ", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

# Cross-validation for Decision Tree
cv_scores_dt = cross_val_score(dt_classifier, X_scaled, y, cv=10, scoring='accuracy')
print("Decision Tree Cross-validation Accuracy: ", cv_scores_dt.mean())  # Mean accuracy across folds

# Optional: Hyperparameter tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
print(f"Best parameters for Decision Tree: {grid_search.best_params_}")

# Train the model with the best parameters
best_dt_classifier = grid_search.best_estimator_

# Predictions and evaluation with the optimized model
y_pred_best_dt = best_dt_classifier.predict(X_test)
print("Optimized Decision Tree Accuracy: ", accuracy_score(y_test, y_pred_best_dt))
print("Optimized Decision Tree Classification Report:\n", classification_report(y_test, y_pred_best_dt))

# Optional: Visualize the Decision Tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(best_dt_classifier, filled=True, feature_names=X.columns, class_names=['Not Donated', 'Donated'], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
