import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'transfusion.csv'
df = pd.read_csv(file_path)

# 1. Check missing values, duplicates, and data types
print("Missing Values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print("Data Types:\n", df.dtypes)
print("Summary Statistics:\n", df.describe())

# 2. Handle missing values (if any, although none were found earlier)
df = df.fillna(df.mean(numeric_only=True))
df = df.dropna()  # Drop rows with non-numeric missing values

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Rename columns for clarity
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

# 5. Remove outliers based on IQR for numeric columns
numeric_cols = ['Recency', 'Frequency', 'Monetary', 'Time']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

# 6. Save cleaned data to a CSV
cleaned_file_path = 'cleaned_transfusion.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}")

# 7. Analysis: Average values by target label
avg_values_by_target = df.groupby('Donated_March_2007').mean()
print("Average Values by Target:\n", avg_values_by_target)

# 8. Plot histogram for one numeric column (e.g., 'Monetary')
df['Monetary'].hist(bins=20)
plt.title('Distribution of Monetary Donations')
plt.xlabel('Monetary (c.c. blood)')
plt.ylabel('Frequency')
plt.show()

