"""Classification metrics computed on held-out data."""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'f1': float(f1_score(y_test, predictions)),
        'roc_auc': float(roc_auc_score(y_test, probabilities)),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
    }
