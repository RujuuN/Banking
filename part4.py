"""Part 4: Model-choice and architecture memo."""
import math
import numpy as np
from common import (
    load_data,
    write_results_file,
    write_csv_file,
    calculate_confusion_metrics,
    get_stratified_split
)
from part2 import train_logistic_regression, knn_predict, sigmoid


def entropy_gini(y):
    """Compute binary entropy and Gini impurity for a label array."""
    n = len(y)
    if n == 0:
        return 0.0, 0.0
    p1 = np.sum(y == 1) / n
    p0 = 1.0 - p1

    ent = 0.0
    if p1 > 0.0:
        ent -= p1 * math.log2(p1)
    if p0 > 0.0:
        ent -= p0 * math.log2(p0)

    gini = 1.0 - (p1 ** 2 + p0 ** 2)
    return ent, gini


def run_part4():
    data = load_data()
    X = data["X"]
    y = data["y"]

    train_idx, test_idx = get_stratified_split(y)
    X_train_raw = X[train_idx]
    y_train = y[train_idx]
    X_test_raw = X[test_idx]
    y_test = y[test_idx]

    # Standardised features from Part 2
    train_mean = np.mean(X_train_raw, axis=0)
    train_sd = np.std(X_train_raw, axis=0, ddof=1)
    X_train_std = (X_train_raw - train_mean) / train_sd
    X_test_std = (X_test_raw - train_mean) / train_sd

    n_train = len(y_train)

    # 4.1 Candidate splits on 560 training rows in original units
    # S1: X6 = 1 (column 5)
    # S2: X7 = 1 (column 6)
    # S3: X3 > 0.5 (column 2)
    # S4: X1 <= 24 (column 0)
    root_ent, root_gini = entropy_gini(y_train)

    splits = [
        ("S1", X_train_raw[:, 5] == 1.0),
        ("S2", X_train_raw[:, 6] == 1.0),
        ("S3", X_train_raw[:, 2] > 0.5),
        ("S4", X_train_raw[:, 0] <= 24.0),
    ]

    split_csv_rows = []
    best_ig = -1.0
    best_ig_name = None
    best_gini_gain = -1.0
    best_gini_name = None
    best_split_condition = None

    for name, mask_yes in splits:
        mask_no = ~mask_yes
        y_yes = y_train[mask_yes]
        y_no = y_train[mask_no]

        n_yes = len(y_yes)
        pos_yes = int(np.sum(y_yes == 1))
        n_no = len(y_no)
        pos_no = int(np.sum(y_no == 1))

        ent_yes, gini_yes = entropy_gini(y_yes)
        ent_no, gini_no = entropy_gini(y_no)

        info_gain = root_ent - ((n_yes / n_train) * ent_yes + (n_no / n_train) * ent_no)
        gini_gain = root_gini - ((n_yes / n_train) * gini_yes + (n_no / n_train) * gini_no)

        split_csv_rows.append([name, n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain])

        if info_gain > best_ig:
            best_ig = info_gain
            best_ig_name = name
            best_split_condition = name

        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_gini_name = name

    write_csv_file(
        "part4_splits.csv",
        ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
        split_csv_rows
    )

    # 4.2 Stump and Voting Ensembles
    # Best split condition: S3 is X3 > 0.5 (or whichever has highest info gain)
    if best_split_condition == "S1":
        train_yes_mask = (X_train_raw[:, 5] == 1.0)
        test_yes_mask = (X_test_raw[:, 5] == 1.0)
    elif best_split_condition == "S2":
        train_yes_mask = (X_train_raw[:, 6] == 1.0)
        test_yes_mask = (X_test_raw[:, 6] == 1.0)
    elif best_split_condition == "S3":
        train_yes_mask = (X_train_raw[:, 2] > 0.5)
        test_yes_mask = (X_test_raw[:, 2] > 0.5)
    elif best_split_condition == "S4":
        train_yes_mask = (X_train_raw[:, 0] <= 24.0)
        test_yes_mask = (X_test_raw[:, 0] <= 24.0)
    else:
        raise ValueError(f"Unknown split: {best_split_condition}")

    stump_rate_yes = float(np.mean(y_train[train_yes_mask]))
    stump_rate_no = float(np.mean(y_train[~train_yes_mask]))

    # Predictions on test set (240 rows):
    # 1. Logistic (Task 2.2 model)
    w_log, b_log, _ = train_logistic_regression(X_train_std, y_train, eta=0.1, epochs=1000)
    test_z = X_test_std @ w_log + b_log
    prob_logistic = sigmoid(test_z)
    pred_logistic = (prob_logistic >= 0.3).astype(int)

    # 2. KNN (best K = 5 from Part 2)
    best_k = 5
    pred_knn, prob_knn = knn_predict(X_train_std, y_train, X_test_std, best_k)

    # 3. Stump: predicted probability is stump_rate_yes if test_yes_mask else stump_rate_no
    prob_stump = np.where(test_yes_mask, stump_rate_yes, stump_rate_no)
    pred_stump = (prob_stump >= 0.3).astype(int)

    # 4. Hard vote: positive when at least two of the three above predict positive
    pred_hard = ((pred_logistic + pred_knn + pred_stump) >= 2).astype(int)

    # 5. Soft vote: positive when mean of 3 probabilities >= 0.3
    prob_soft = (prob_logistic + prob_knn + prob_stump) / 3.0
    pred_soft = (prob_soft >= 0.3).astype(int)

    # Evaluate all 5 models on test set
    models_to_eval = [
        ("logistic", pred_logistic),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard),
        ("soft_vote", pred_soft),
    ]

    voting_csv_rows = []
    for model_name, preds in models_to_eval:
        m = calculate_confusion_metrics(y_test, preds)
        voting_csv_rows.append([
            model_name,
            m["tp"],
            m["fp"],
            m["tn"],
            m["fn"],
            m["precision"],
            m["recall"],
            m["f1"]
        ])

    write_csv_file(
        "part4_voting.csv",
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_csv_rows
    )

    # 4.3 Two rounds of AdaBoost by hand
    # Point 1..8
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    adaboost_csv_rows = []

    # Round 1
    w1 = np.full(8, 1.0 / 8.0)
    err1 = float(np.sum(w1[h1 != y_ada]))
    alpha1 = 0.5 * math.log((1.0 - err1) / err1)
    w1_unnorm = w1 * np.exp(-alpha1 * y_ada * h1)
    w1_after = w1_unnorm / np.sum(w1_unnorm)

    for pt in range(8):
        adaboost_csv_rows.append([1, pt + 1, w1[pt], w1_unnorm[pt], w1_after[pt]])

    # Round 2
    w2 = w1_after.copy()
    err2 = float(np.sum(w2[h2 != y_ada]))
    alpha2 = 0.5 * math.log((1.0 - err2) / err2)
    w2_unnorm = w2 * np.exp(-alpha2 * y_ada * h2)
    w2_after = w2_unnorm / np.sum(w2_unnorm)

    for pt in range(8):
        adaboost_csv_rows.append([2, pt + 1, w2[pt], w2_unnorm[pt], w2_after[pt]])

    write_csv_file(
        "part4_adaboost.csv",
        ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        adaboost_csv_rows
    )

    # Ensemble accuracy
    ensemble_score = alpha1 * h1 + alpha2 * h2
    ensemble_preds = np.where(ensemble_score >= 0, 1, -1)
    ensemble_acc = float(np.mean(ensemble_preds == y_ada))

    # 4.4 Out-of-bag share
    oob_fraction = (1.0 - 1.0 / 560.0) ** 560

    # 4.5 Neural Network Architecture Calculations
    # Model A Option 1: MLP 784 -> 16 -> 16 -> 10
    mlp_h1 = 784 * 16 + 16
    mlp_h2 = 16 * 16 + 16
    mlp_out = 16 * 10 + 10
    mlp_total = mlp_h1 + mlp_h2 + mlp_out

    # Model A Option 2: CNN
    # cnn_conv1: 5x5 filters, 6 filters, stride 1, padding 2. Input 28x28x1 -> Output 28x28x6
    cnn_conv1 = 5 * 5 * 1 * 6 + 6  # 156
    # Max-pool 2x2 stride 2 -> 14x14x6, 0 params
    # cnn_conv2: 5x5 filters, 16 filters, stride 1, padding 0. Input 14x14x6 -> Output 10x10x16
    cnn_conv2 = 5 * 5 * 6 * 16 + 16  # 2416
    # Max-pool 2x2 stride 2 -> 5x5x16 = 400 flattened, 0 params
    # cnn_dense1: 400 -> 32
    cnn_dense1 = 400 * 32 + 32  # 12832
    # cnn_dense2: 32 -> 10
    cnn_dense2 = 32 * 10 + 10  # 330
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    arch_rows = [
        ["mlp_hidden1", mlp_h1],
        ["mlp_hidden2", mlp_h2],
        ["mlp_output", mlp_out],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2],
    ]
    write_csv_file("part4_architecture.csv", ["layer", "parameters"], arch_rows)

    # Scaling comparison on 128x128x3:
    fc_layer_params = 128 * 128 * 3 * 1024 + 1024  # 50,332,672
    conv_layer_params = 3 * 3 * 3 * 64 + 64  # 1,792

    # Model B: RNN
    # X_t: 64, H_t: 128, Y: 16
    # W_xh: 128 x 64, W_hh: 128 x 128, b_h: 128
    # W_hy: 16 x 128, b_y: 16
    rnn_total = 128 * 64 + 128 * 128 + 128 + 16 * 128 + 16  # 26,768

    results = [
        ("root_entropy", root_ent),
        ("root_gini", root_gini),
        ("best_split_by_info_gain", best_ig_name),
        ("best_split_by_gini_gain", best_gini_name),
        ("stump_rate_yes", stump_rate_yes),
        ("stump_rate_no", stump_rate_no),
        ("round1_error", err1),
        ("round1_alpha", alpha1),
        ("round2_error", err2),
        ("round2_alpha", alpha2),
        ("ensemble_accuracy", ensemble_acc),
        ("oob_fraction", oob_fraction),
        ("mlp_total", mlp_total),
        ("cnn_total", cnn_total),
        ("fc_layer_params", fc_layer_params),
        ("conv_layer_params", conv_layer_params),
        ("rnn_total", rnn_total),
    ]

    write_results_file("part4_results.txt", results)
    return results


if __name__ == "__main__":
    run_part4()
