
import pandas as pd

FILE_PATH = "/mnt/data/filled_healthcare_appointments.csv"

def load_data():
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully")
    print(df.head())
    print(df.columns)
    return df

if __name__ == "__main__":
    load_data()
