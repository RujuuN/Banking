"""Part 2: Classification and evaluation."""
import math
import numpy as np
from common import (
    load_data,
    write_results_file,
    write_csv_file,
    calculate_confusion_metrics,
    get_stratified_split
)


def sigmoid(z):
    """Numerically stable sigmoid function."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def compute_bce_loss(y, p):
    """Binary cross entropy loss."""
    eps = 1e-15
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))


def train_logistic_regression(X_train, y_train, eta=0.1, epochs=1000, record_epochs=(0, 1, 10, 100, 1000)):
    """Train logistic regression from w=0, b=0 using full batch gradient descent."""
    n, p = X_train.shape
    w = np.zeros(p, dtype=np.float64)
    b = 0.0
    history = {}

    for step in range(epochs + 1):
        z = X_train @ w + b
        y_hat = sigmoid(z)
        loss = compute_bce_loss(y_train, y_hat)

        if step in record_epochs:
            history[step] = loss

        if step < epochs:
            # Gradients: dL/dw = (1/n) X^T (y_hat - y), dL/db = (1/n) sum(y_hat - y)
            diff = y_hat - y_train
            grad_w = (X_train.T @ diff) / n
            grad_b = float(np.sum(diff)) / n

            w -= eta * grad_w
            b -= eta * grad_b

    return w, b, history


def knn_predict(X_ref, y_ref, X_query, k):
    """
    Predict KNN for queries based on reference data.
    Distance: Euclidean distance.
    Tie-breaking: position in file order among reference rows (stable sort).
    Returns binary predictions (1 if >= (k+1)/2 positives) and positive fraction.
    """
    n_query = len(X_query)
    preds = np.zeros(n_query, dtype=int)
    pos_fractions = np.zeros(n_query, dtype=np.float64)

    for i in range(n_query):
        q = X_query[i]
        dists = np.sum((X_ref - q) ** 2, axis=1)  # squared Euclidean preserves rank
        nn_indices = np.argsort(dists, kind="stable")[:k]
        pos_count = int(np.sum(y_ref[nn_indices]))
        pos_fractions[i] = pos_count / k
        preds[i] = 1 if pos_count >= (k + 1) // 2 else 0

    return preds, pos_fractions


def run_part2():
    data = load_data()
    X = data["X"]
    y = data["y"]

    # 2.1 Split and standardise
    train_idx, test_idx = get_stratified_split(y)
    train_rows = len(train_idx)
    test_rows = len(test_idx)
    train_positives = int(np.sum(y[train_idx] == 1))
    test_positives = int(np.sum(y[test_idx] == 1))

    X_train_raw = X[train_idx]
    y_train = y[train_idx]
    X_test_raw = X[test_idx]
    y_test = y[test_idx]

    # Standardise X1 to X7 with training mean and sample SD (ddof=1)
    train_mean = np.mean(X_train_raw, axis=0)
    train_sd = np.std(X_train_raw, axis=0, ddof=1)

    X_train = (X_train_raw - train_mean) / train_sd
    X_test = (X_test_raw - train_mean) / train_sd

    # 2.2 Logistic regression from scratch (eta=0.1, 1000 epochs)
    w_final, b_final, loss_hist = train_logistic_regression(
        X_train, y_train, eta=0.1, epochs=1000, record_epochs=(0, 1, 10, 100, 1000)
    )

    # Test BCE
    test_z = X_test @ w_final + b_final
    test_p = sigmoid(test_z)
    test_bce = compute_bce_loss(y_test, test_p)

    # 2.3 Learning rate effect (200 updates with eta in {0.01, 0.1, 1.0})
    _, _, hist_001 = train_logistic_regression(X_train, y_train, eta=0.01, epochs=200, record_epochs=(200,))
    _, _, hist_01 = train_logistic_regression(X_train, y_train, eta=0.1, epochs=200, record_epochs=(200,))
    _, _, hist_10 = train_logistic_regression(X_train, y_train, eta=1.0, epochs=200, record_epochs=(200,))

    loss_epoch_200_eta_001 = hist_001[200]
    loss_epoch_200_eta_01 = hist_01[200]
    loss_epoch_200_eta_10 = hist_10[200]

    # 2.4 Odds ratios: exp(weight)
    odds_ratios = np.exp(w_final)

    # 2.5 Evaluate on test rows at tau = 0.5 and tau = 0.3
    pred_05 = (test_p >= 0.5).astype(int)
    m_05 = calculate_confusion_metrics(y_test, pred_05)

    pred_03 = (test_p >= 0.3).astype(int)
    m_03 = calculate_confusion_metrics(y_test, pred_03)

    blind_accuracy = float(np.sum(y_test == 0) / len(y_test))

    # 2.6 Threshold sweep
    threshold_rows = []
    for tau_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_tau = (test_p >= tau_val).astype(int)
        m_t = calculate_confusion_metrics(y_test, pred_tau)
        threshold_rows.append([
            tau_val,
            m_t["tp"],
            m_t["fp"],
            m_t["tn"],
            m_t["fn"],
            m_t["recall"],  # tpr
            m_t["fpr"]
        ])
    write_csv_file("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], threshold_rows)

    # 2.7 K-nearest neighbours with 5-fold CV
    # Folds: 0-based position in training set mod 5
    folds = np.arange(train_rows) % 5
    k_values = list(range(1, 23, 2))  # 1, 3, 5, ..., 21
    knn_cv_rows = []
    best_k = None
    best_f1 = -1.0

    for k in k_values:
        fold_f1s = []
        for fold in range(5):
            val_mask = (folds == fold)
            ref_mask = ~val_mask

            X_ref = X_train[ref_mask]
            y_ref = y_train[ref_mask]
            X_val = X_train[val_mask]
            y_val = y_train[val_mask]

            preds_val, _ = knn_predict(X_ref, y_ref, X_val, k)
            m_val = calculate_confusion_metrics(y_val, preds_val)
            fold_f1s.append(m_val["f1"])

        mean_f1 = float(np.mean(fold_f1s))
        knn_cv_rows.append([k, mean_f1])
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k = k

    write_csv_file("part2_knn_cv.csv", ["K", "mean_cv_f1"], knn_cv_rows)

    # Refit with all 560 training rows and evaluate on 240 test rows
    knn_test_preds, _ = knn_predict(X_train, y_train, X_test, best_k)
    m_knn = calculate_confusion_metrics(y_test, knn_test_preds)

    # 2.8 Choose the alert threshold with money: cost = 50 * fp + 400 * fn
    train_z = X_train @ w_final + b_final
    train_p = sigmoid(train_z)

    best_cost_train = float("inf")
    best_tau = None

    for k in range(1, 20):
        tau = k / 20.0
        train_pred_tau = (train_p >= tau).astype(int)
        fp_train = int(np.sum((train_pred_tau == 1) & (y_train == 0)))
        fn_train = int(np.sum((train_pred_tau == 0) & (y_train == 1)))
        cost_train = 50 * fp_train + 400 * fn_train

        # Strictly lower cost (ties go to smaller tau)
        if cost_train < best_cost_train:
            best_cost_train = cost_train
            best_tau = tau

    cost_tau = best_tau
    cost_train_at_tau = best_cost_train

    # Test evaluations
    test_pred_tau = (test_p >= cost_tau).astype(int)
    fp_test_tau = int(np.sum((test_pred_tau == 1) & (y_test == 0)))
    fn_test_tau = int(np.sum((test_pred_tau == 0) & (y_test == 1)))
    cost_test_at_tau = 50 * fp_test_tau + 400 * fn_test_tau

    fp_test_05 = int(np.sum((pred_05 == 1) & (y_test == 0)))
    fn_test_05 = int(np.sum((pred_05 == 0) & (y_test == 1)))
    cost_test_at_05 = 50 * fp_test_05 + 400 * fn_test_05

    cost_test_flag_nobody = 400 * test_positives

    # Collect Part 2 results
    results = [
        ("train_rows", train_rows),
        ("train_positives", train_positives),
        ("test_rows", test_rows),
        ("test_positives", test_positives),
        ("loss_epoch_0", loss_hist[0]),
        ("loss_epoch_1", loss_hist[1]),
        ("loss_epoch_10", loss_hist[10]),
        ("loss_epoch_100", loss_hist[100]),
        ("loss_epoch_1000", loss_hist[1000]),
        ("weight_X1", w_final[0]),
        ("weight_X2", w_final[1]),
        ("weight_X3", w_final[2]),
        ("weight_X4", w_final[3]),
        ("weight_X5", w_final[4]),
        ("weight_X6", w_final[5]),
        ("weight_X7", w_final[6]),
        ("bias", b_final),
        ("test_bce", test_bce),
        ("loss_epoch_200_eta_0.01", loss_epoch_200_eta_001),
        ("loss_epoch_200_eta_0.1", loss_epoch_200_eta_01),
        ("loss_epoch_200_eta_1.0", loss_epoch_200_eta_10),
        ("odds_ratio_X1", odds_ratios[0]),
        ("odds_ratio_X2", odds_ratios[1]),
        ("odds_ratio_X3", odds_ratios[2]),
        ("odds_ratio_X4", odds_ratios[3]),
        ("odds_ratio_X5", odds_ratios[4]),
        ("odds_ratio_X6", odds_ratios[5]),
        ("odds_ratio_X7", odds_ratios[6]),
        ("tau_0.5_tp", m_05["tp"]),
        ("tau_0.5_fp", m_05["fp"]),
        ("tau_0.5_tn", m_05["tn"]),
        ("tau_0.5_fn", m_05["fn"]),
        ("tau_0.5_accuracy", m_05["accuracy"]),
        ("tau_0.5_precision", m_05["precision"]),
        ("tau_0.5_recall", m_05["recall"]),
        ("tau_0.5_f1", m_05["f1"]),
        ("tau_0.5_specificity", m_05["specificity"]),
        ("tau_0.3_tp", m_03["tp"]),
        ("tau_0.3_fp", m_03["fp"]),
        ("tau_0.3_tn", m_03["tn"]),
        ("tau_0.3_fn", m_03["fn"]),
        ("tau_0.3_accuracy", m_03["accuracy"]),
        ("tau_0.3_precision", m_03["precision"]),
        ("tau_0.3_recall", m_03["recall"]),
        ("tau_0.3_f1", m_03["f1"]),
        ("tau_0.3_specificity", m_03["specificity"]),
        ("blind_accuracy", blind_accuracy),
        ("knn_best_k", best_k),
        ("knn_tp", m_knn["tp"]),
        ("knn_fp", m_knn["fp"]),
        ("knn_tn", m_knn["tn"]),
        ("knn_fn", m_knn["fn"]),
        ("knn_accuracy", m_knn["accuracy"]),
        ("knn_precision", m_knn["precision"]),
        ("knn_recall", m_knn["recall"]),
        ("knn_f1", m_knn["f1"]),
        ("cost_tau", cost_tau),
        ("cost_train_at_tau", cost_train_at_tau),
        ("cost_test_at_tau", cost_test_at_tau),
        ("cost_test_at_0.5", cost_test_at_05),
        ("cost_test_flag_nobody", cost_test_flag_nobody),
    ]

    write_results_file("part2_results.txt", results)
    return {
        "w_final": w_final,
        "b_final": b_final,
        "train_mean": train_mean,
        "train_sd": train_sd,
        "best_k": best_k,
        "results": results
    }


if __name__ == "__main__":
    run_part2()
