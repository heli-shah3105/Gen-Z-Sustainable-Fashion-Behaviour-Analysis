import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load CSV
csv_path = "PGM GOOGLE FORM RESPONSES.csv"
df_raw = pd.read_csv(csv_path)

print(f"Dataset shape: {df_raw.shape}")
print("\nColumns:")
for i, col in enumerate(df_raw.columns):
    print(f"[{i}] {col.strip()}")

# Rename columns to clean, standardized variable names
column_mapping = {
    df_raw.columns[0]: 'Timestamp',
    df_raw.columns[1]: 'Purchase_Frequency',
    df_raw.columns[2]: 'Monthly_Spending',
    df_raw.columns[3]: 'Purchase_Channel',
    df_raw.columns[4]: 'Awareness',
    df_raw.columns[5]: 'Attitude',
    df_raw.columns[6]: 'Social_Influence',
    df_raw.columns[7]: 'PBC',
    df_raw.columns[8]: 'Price_Sensitivity',
    df_raw.columns[9]: 'Trust',
    df_raw.columns[10]: 'Intention',
    df_raw.columns[11]: 'Sustainable_Preference',
    df_raw.columns[12]: 'Willingness_To_Pay',
    df_raw.columns[13]: 'Behaviour'
}

df = df_raw.rename(columns=column_mapping)
print("\nRenamed Columns:", df.columns.tolist())
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nBehaviour counts:\n", df['Behaviour'].value_counts())
print("\nWTP vs Price Sensitivity correlation:")
corr = df[['Price_Sensitivity', 'Willingness_To_Pay', 'Awareness', 'Attitude', 'Social_Influence', 'PBC', 'Trust', 'Intention', 'Sustainable_Preference']].corr()
print(corr)
