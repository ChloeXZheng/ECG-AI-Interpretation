# date: 9/21/2026
# function: records experiment results in Excel
from openpyxl import load_workbook
from pathlib import Path
from datetime import datetime
def get_next_experiment_id(workbook):
    """
    Finds the highest existing experiment number
    and returns the next available experiment ID.
    """
    experiment_sheet = workbook["experiment_log"]
    highest_number = 0
    # Skip the header row
    for row in experiment_sheet.iter_rows(
        min_row=1,
        values_only=True
    ):
        experiment_id = row[0]
        if experiment_id is None:
            continue
        # Expected format: EXP001
        if str(experiment_id).startswith("EXP"):
            try:
                number = int(str(experiment_id)[3:])
                highest_number = max(highest_number, number)
            except ValueError:
                pass
    return f"EXP{highest_number + 1:03d}"
def log_experiment(
    file_path,
    task,
    features,
    model_name,
    results
):
    file_path = Path(file_path)
    workbook = load_workbook(file_path)
    experiment_sheet = workbook["experiment_log"]
    fold_sheet = workbook["fold_results"]
    # Automatically generate unique ID
    experiment_id = get_next_experiment_id(workbook)
    # Current date
    date = datetime.now().strftime("%Y-%m-%d")
    # One row for the overall experiment
    experiment_sheet.append([
        experiment_id,
        date,
        task,
        features,
        model_name,
        results["mean_accuracy"],
        results["std_accuracy"],
        results["mean_macro_f1"],
        results["std_macro_f1"],
        results["mean_balanced_accuracy"],
        results["std_balanced_accuracy"]
    ])
    # One row for each fold
    for fold_number, fold_result in enumerate(
        results["fold_results"],
        start=1
    ):
        fold_sheet.append([
            experiment_id,
            fold_number,
            fold_result["accuracy"],
            fold_result["macro_f1"],
            fold_result["balanced_accuracy"]
        ])
    workbook.save(file_path)
    print(f"Saved {experiment_id}: {task} / {model_name}")
    return experiment_id