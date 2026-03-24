import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import MODEL_PATH, MODEL_MAX_ITER, TEST_SIZE


def train_and_evaluate_model(X, y):  # model training and evaluation
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    # training
    logging.info("Training model with Logic regression.")
    model = LogisticRegression(max_iter=MODEL_MAX_ITER,
                               class_weight='balanced')  # scaling by LogisticRegression, iter change for convergence, and we use balanced to handle class imbalance
    model.fit(X_train, y_train)

    # evaluation of the model to the test data
    logging.info("Evaluating model with Logic regression.")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"Model accuracy on test set: {accuracy*100:.2f}")

    # save the model locally
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    return model


def predict_risk(model, X):
    predicted_classes = model.predict(X)  # 0 for no risk (<0,5), 1 for risk (>=0.5)
    probs = model.predict_proba(X)[:, 1]  # the probability of the positive class (risk)

    return predicted_classes, probs


# if needed we can load our previously trained model from the disk
def load_trained_model(MODEL_PATH):
    logging.info("Loading trained model.")
    return joblib.load(MODEL_PATH)