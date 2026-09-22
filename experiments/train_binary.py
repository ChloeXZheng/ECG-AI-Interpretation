# Date: 9/21/2026
# Function: training for binary data

import sys
sys.path.append("src")

import trainfunc as tf
import models
import results


## --- LOAD DATA --- ##

X, y, participant_ids = tf.load_processed_data(
    "data/processed_features.npz"
)


## --- MAKE BINARY DATA --- ##

X, y, participant_ids = tf.make_binary_data(
    X,
    y,
    participant_ids
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


## --- RUN CROSS VALIDATION --- ##

for number, (name, model) in enumerate(
    models_to_test.items(),
    start=1
):

    print("=" * 60)
    print(name)
    print("=" * 60)

    cv_results = tf.evaluate_stratified_group_cv(
        model,
        X,
        y,
        participant_ids,
        n_splits=5,
        random_state=42
    )

    results.log_experiment(
        file_path="data/model_results.xlsx",
        task="binary",
        features="all 84",
        model_name=name,
        results=cv_results
    )