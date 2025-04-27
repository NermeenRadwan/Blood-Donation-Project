import pandas as pd

def clean_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values, duplicates, etc.
    df = df.fillna(df.mean(numeric_only=True))
    df = df.dropna()
    df = df.drop_duplicates()

    # Rename columns
    df.rename(columns={"Recency (months)": "Recency", "Frequency (times)": "Frequency",
                       "Monetary (c.c. blood)": "Monetary", "Time (months)": "Time",
                       "whether he/she donated blood in March 2007": "Donated_March_2007"}, inplace=True)

    return df