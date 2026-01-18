
import pandas as pd
import statsmodels.api as sm

FILE_PATH = "/mnt/data/filled_healthcare_appointments.csv"

df = pd.read_csv(FILE_PATH)

X = df[['Age', 'Scholarship', 'SMS_received']]
y = df['No_show']

X_ols = sm.add_constant(X)

ols_model = sm.OLS(y, X_ols).fit()

print("===== OLS RESULTS =====")
print(ols_model.summary())
