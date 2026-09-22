# date: 9/21/2026
# function: tests that all models can train and make predictions


import numpy as np
import sys

sys.path.append("src")

import models


## --- CREATE TEST DATA --- ##

np.random.seed(42)

X = np.random.randn(100, 84)

y = np.array(
    [0] * 50 +
    [1] * 50
)


## --- MODELS --- ##

models_to_test = {
    "dummy": models.dummy(),
    "logistic_regression": models.logistic_regression(),
    "lda": models.lda(),
    "linear_svm": models.linear_svm(),
    "rbf_svm": models.rbf_svm(),
    "random_forest": models.random_forest(),
    "gradient_boosting": models.gradient_boosting()
}


## --- TEST EACH MODEL --- ##

for name, model in models_to_test.items():

    print(f"Testing {name}...")

    # Train
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    # Check number of predictions
    assert len(predictions) == len(y), (
        f"{name} produced the wrong number of predictions"
    )

    # Check labels are valid
    assert set(np.unique(predictions)).issubset({0, 1}), (
        f"{name} produced unexpected labels"
    )

    print(f"  {name}: passed")


## --- FINAL --- ##

print()
print("All model tests passed!")