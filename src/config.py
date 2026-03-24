import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_FILEPATH = "../data/finance_dataset.csv"
MODEL_PATH = "../models/logistic_model.plk"

# processing parameters
EPSILON = 0.001  # small value to avoid division by zero in ratio calculations
LARGE_TRANSACTION_THRESHOLD = 1000  # threshold for flagging large transactions
QUANTILE_THRESHOLD = 0.9  # quantile threshold for high-risk transactions based on amount

# model parameters
MODEL_MAX_ITER = 100000
TEST_SIZE = 0.20