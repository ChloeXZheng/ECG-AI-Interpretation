# Date: 9/21/2026
# Function: set up all the models

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def dummy():
    return DummyClassifier(strategy="most_frequent")


def logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ])


def lda():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LinearDiscriminantAnalysis())
    ])


def linear_svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="linear"
        ))
    ])


def rbf_svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf"
        ))
    ])


def random_forest():
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )


def gradient_boosting():
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )