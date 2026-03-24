import pandas as pd
import numpy as np
from config import EPSILON, LARGE_TRANSACTION_THRESHOLD, QUANTILE_THRESHOLD


def preprocess_data(df: pd.DataFrame):
    data = df.copy()

    data['amount_to_balance_ratio'] = data['Transaction_Amount'] / (
            data['Account_Balance'] + EPSILON)  # +00.1 to avoid div by 0

    data['transaction_amount_log'] = np.log1p(
        data['Transaction_Amount'])  # using log(x+1) for better results {avoid extreme values etc.}

    # flags
    data['is_negative_balance'] = (data['Account_Balance'] < 0).astype(int)
    data['is_large_transaction'] = (data['Transaction_Amount'] > LARGE_TRANSACTION_THRESHOLD).astype(int)

    # conditions for target_risk
    cond1 = (data['Transaction_Amount'] > data['Transaction_Amount'].quantile(QUANTILE_THRESHOLD))
    cond2 = (data['Account_Balance'] < 0)
    cond3 = (data['Transaction_Status'] != 'Completed')

    data['target_risk'] = np.where(cond1 | cond2 | cond3, 1, 0)

    data = pd.get_dummies(data, columns=['Transaction_Type', 'Payment_Method', 'Category', 'Location'])

    # time related data formating
    data['transaction_day'] = data['Date'].dt.day
    data['transaction_month'] = data['Date'].dt.month

    # Scaling for accuracy in training
    X = data.drop(columns=['target_risk', 'Customer_ID', 'Date', 'Transaction_Status'])
    y = data['target_risk']

    return X, y