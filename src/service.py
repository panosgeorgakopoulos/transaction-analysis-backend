import logging
import os
from data_loader import load_transactions
from preprocessing import preprocess_data
from model import train_and_evaluate_model, predict_risk, load_trained_model
from config import MODEL_PATH
from src.config import DATA_FILEPATH


def main():
    logging.info("Starting the financial transaction risk classification process.")

    df = load_transactions(DATA_FILEPATH)
    X, y = preprocess_data(df)
    if not os.path.exists(MODEL_PATH):
        logging.info("Pre-trained model not found. Training begins...")
        model = train_and_evaluate_model(X, y)
    else:
        logging.info("Pre-trained model found. Loading model...")
        model = load_trained_model(MODEL_PATH)


    classes, scores = predict_risk(model, X)

    print('Total transactions analyzed: ', len(X))
    print((classes.sum() / len(classes)) * 100, "% of transactions are classified as high-risk.")
    print("Sample predictions:")
    for i in range(5):  # print first 5 predictions
        print(f"Transaction {i + 1}: Risk Class = {classes[i]}, Risk Score = {(scores[i])*100:.2f}%")
    print("\nModel training and prediction completed.")


if __name__ == "__main__":
    main()
