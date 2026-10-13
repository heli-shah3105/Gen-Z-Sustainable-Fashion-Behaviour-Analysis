import pandas as pd
import numpy as np

df = pd.read_csv("PGM GOOGLE FORM RESPONSES.csv")
print("Dataframe shape:", df.shape)
print("Raw columns:", df.columns.tolist())
print("\nFirst 3 rows:\n", df.iloc[:3])
print("\nUnique values in last column:\n", df.iloc[:, -1].value_counts())
