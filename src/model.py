from sklearn.linear_model import LogisticRegression


def train_model(X, y):
    model = LogisticRegression(max_iter=100000,
                               class_weight='balanced')  # scaling by LogisticRegression, iter change for convergence, and we use balanced to handle class imbalance
    model.fit(X, y)  # model training
    return model


def predict_risk(model, X):
    predicted_classes = model.predict(X)  # 0 for no risk (<0,5), 1 for risk (>=0.5)
    probs = model.predict_proba(X)[:, 1]  # the probability of the positive class (risk)

    return predicted_classes, probs
