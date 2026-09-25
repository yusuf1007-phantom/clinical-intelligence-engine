from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def build_model(name: str, params: dict, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "linear_svm":
        return LinearSVC(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown model: {name}")
