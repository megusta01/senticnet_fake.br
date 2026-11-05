"""Reproducible modeling setup based on lr_rf.ipynb."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def make_searches(*, n_jobs: int = -1) -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                                   class_weight='balanced')),
    ])
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE,
                                      class_weight='balanced')),
    ])
    return {
        'logistic_regression': GridSearchCV(
            lr, {'clf__C': [0.01, 0.1, 1, 10]},
            cv=cv, scoring='f1', n_jobs=n_jobs),
        'random_forest': GridSearchCV(
            rf, {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 10, 20]},
            cv=cv, scoring='f1', n_jobs=n_jobs),
    }
