from data_loader import load_transactions
from preprocessing import preprocess_data
from model import train_model, predict_risk


def main():
    filepath = '../data/finance_dataset.csv'  # path to our dataset
    df = load_transactions(filepath)
    X, y = preprocess_data(df)
    model = train_model(X, y)
    classes, scores = predict_risk(model, X)
    print('Dataset size: ', len(df), 'records')
    print((classes.sum() / len(classes)) * 100, "% of transactions are classified as risky.")
    for i in range(5):  # print first 5 predictions
        print(f"Transaction {i + 1}: Risk Class = {classes[i]}, Risk Score = {(scores[i])*100:.2f}%")
    print("\nModel training and prediction completed.")


if __name__ == "__main__":
    main()
