"""Shared helper functions and constants for the capstone project."""
import csv
import math
import os
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "records.csv")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_data(filepath=DATA_PATH):
    """Load records.csv and return structured dictionary of arrays."""
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    record_ids = [r[0] for r in rows]
    # Columns:
    # 0: record_id
    # 1: account_age_months (X1)
    # 2: failed_txn_count_90d (X2)
    # 3: credit_utilisation_ratio (X3)
    # 4: avg_monthly_inflow_inr_thousands (X4)
    # 5: repayment_score (X5)
    # 6: irregular_salary_flag (X6)
    # 7: autopay_flag (X7)
    # 8: loan_default (outcome)
    # 9: monthly_spend_inr_thousands (value)
    data_floats = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float64)
    X = data_floats[:, 0:7]  # X1 to X7
    y = data_floats[:, 7].astype(int)  # outcome
    value = data_floats[:, 8]  # value

    return {
        "record_ids": record_ids,
        "X": X,
        "y": y,
        "value": value,
        "header": header
    }


def format_value(val):
    """Format value for results files: 4 decimals for floats, ints as ints, yes/no for bools."""
    if isinstance(val, (bool, np.bool_)):
        return "yes" if val else "no"
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    return str(val)


def write_results_file(filename, kv_pairs):
    """Write key = value pairs to outputs/filename."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUTS_DIR, filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        for k, v in kv_pairs:
            f.write(f"{k} = {format_value(v)}\n")
    print(f"Wrote {filepath} ({len(kv_pairs)} keys)")


def write_csv_file(filename, header, rows):
    """Write CSV file with comma delimiter and 4 decimal formatting for floats."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUTS_DIR, filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            formatted_row = [format_value(x) for x in row]
            writer.writerow(formatted_row)
    print(f"Wrote {filepath} ({len(rows)} rows)")


def calculate_confusion_metrics(y_true, y_pred):
    """Calculate confusion matrix cells and evaluation metrics."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1
    }


def get_stratified_split(y):
    """
    Perform stratified 70/30 split according to Task 2.1 rules:
    Outcome 0: first floor(7 * count0 / 10) in file order to train, rest to test.
    Outcome 1: first floor(7 * count1 / 10) in file order to train, rest to test.
    Both sets keep file order.
    """
    idx_0 = [i for i, val in enumerate(y) if val == 0]
    idx_1 = [i for i, val in enumerate(y) if val == 1]

    n_train_0 = math.floor(7 * len(idx_0) / 10)
    n_train_1 = math.floor(7 * len(idx_1) / 10)

    train_idx_0 = idx_0[:n_train_0]
    test_idx_0 = idx_0[n_train_0:]

    train_idx_1 = idx_1[:n_train_1]
    test_idx_1 = idx_1[n_train_1:]

    train_indices = sorted(train_idx_0 + train_idx_1)
    test_indices = sorted(test_idx_0 + test_idx_1)

    return train_indices, test_indices
