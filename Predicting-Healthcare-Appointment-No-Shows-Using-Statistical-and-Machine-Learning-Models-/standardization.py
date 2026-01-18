import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/mnt/data/filled_healthcare_appointments.csv")

features = ['Age', 'Scholarship', 'SMS_received']

scaler = StandardScaler()

df_standardized = df.copy()
df_standardized[features] = scaler.fit_transform(df[features])

print("===== STANDARDIZED DATA (FIRST 5 ROWS) =====")
print(df_standardized[features].head())

df_standardized.to_csv("/mnt/data/standardized_healthcare_appointments.csv", index=False)
print("Standardized file saved.")
