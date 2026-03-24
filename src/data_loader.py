import logging
import pandas as pd
from config import DATA_FILEPATH


def load_transactions(filepath: str = DATA_FILEPATH) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    required_columns = [
        'Date', 'Transaction_Amount', 'Customer_ID', 'Account_Balance',
        'Transaction_Type', 'Payment_Method', 'Category', 'Location'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Λείπουν οι υποχρεωτικές στήλες: {missing_columns}")

    df['Transaction_Amount'] = pd.to_numeric(df['Transaction_Amount'], errors='coerce')
    df['Account_Balance'] = pd.to_numeric(df['Account_Balance'], errors='coerce')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Transaction_Amount', 'Customer_ID'])

    logging.info(f" {len(df)} records have been loaded from the file {filepath}")

    return df
