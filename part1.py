"""Part 1: Estimation, inference, and a regression baseline."""
import math
import numpy as np
from common import load_data, write_results_file


def run_part1():
    data = load_data()
    X = data["X"]  # X1..X7
    # Column mapping:
    # 0: X1 (account_age_months)
    # 1: X2 (failed_txn_count_90d)
    # 2: X3 (credit_utilisation_ratio)
    # 3: X4 (avg_monthly_inflow_inr_thousands)
    # 4: X5 (repayment_score)
    # 5: X6 (irregular_salary_flag)
    # 6: X7 (autopay_flag)
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    X5 = X[:, 4]
    X6 = X[:, 5]
    X7 = X[:, 6]
    value = data["value"]

    # 1.1 Estimate with an interval
    n = len(value)
    mean_value = float(np.mean(value))
    sd_value = float(np.std(value, ddof=1))
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # 1.2 Test a planning claim: H0: mu = 8.0 vs H1: mu != 8.0
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # 1.3 Size a future sample: E = 0.5
    E = 0.5
    n_required = math.ceil((1.96 * sd_value / E) ** 2)

    # 1.4 Simple regression by closed form: value on X4
    # cov(X4, value) / var(X4) with divisor n - 1
    mean_X4 = float(np.mean(X4))
    cov_X4_val = float(np.sum((X4 - mean_X4) * (value - mean_value)) / (n - 1))
    var_X4 = float(np.sum((X4 - mean_X4) ** 2) / (n - 1))
    reg_slope = cov_X4_val / var_X4
    reg_intercept = mean_value - reg_slope * mean_X4

    fitted = reg_intercept + reg_slope * X4
    tss = float(np.sum((value - mean_value) ** 2))
    ess = float(np.sum((fitted - mean_value) ** 2))
    rss = float(np.sum((value - fitted) ** 2))
    tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(value - fitted)))

    # 1.5 Gradient descent
    sd_X4 = math.sqrt(var_X4)
    z = (X4 - mean_X4) / sd_X4

    m = 0.0
    c = 0.0
    eta = 0.1
    gd_costs = {}

    for step in range(501):
        y_hat = m * z + c
        e = value - y_hat
        cost = float(np.mean(e ** 2))  # (1/n) sum e^2
        if step in (0, 1, 10, 100, 500):
            gd_costs[step] = cost

        if step < 500:
            grad_m = -(2.0 / n) * float(np.sum(z * e))
            grad_c = -(2.0 / n) * float(np.sum(e))
            m -= eta * grad_m
            c -= eta * grad_c

    gd_slope = m / sd_X4
    gd_intercept = c - m * mean_X4 / sd_X4
    gd_slope_gap = abs(gd_slope - reg_slope)

    # 1.6 Funnel check and log remedy
    # Sort rows by fitted values of Task 1.4 ascending, stable sort
    sort_order_raw = np.argsort(fitted, kind="stable")
    resid_raw = value - fitted
    resid_lower_raw = resid_raw[sort_order_raw[:400]]
    resid_upper_raw = resid_raw[sort_order_raw[400:]]
    resid_sd_lower_raw = float(np.std(resid_lower_raw, ddof=1))
    resid_sd_upper_raw = float(np.std(resid_upper_raw, ddof=1))
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    # Log-log regression: ln(value) on ln(X4)
    ln_val = np.log(value)
    ln_X4 = np.log(X4)
    mean_ln_val = float(np.mean(ln_val))
    mean_ln_X4 = float(np.mean(ln_X4))

    cov_ln = float(np.sum((ln_X4 - mean_ln_X4) * (ln_val - mean_ln_val)) / (n - 1))
    var_ln_X4 = float(np.sum((ln_X4 - mean_ln_X4) ** 2) / (n - 1))
    loglog_slope = cov_ln / var_ln_X4
    loglog_intercept = mean_ln_val - loglog_slope * mean_ln_X4

    fitted_log = loglog_intercept + loglog_slope * ln_X4
    tss_log = float(np.sum((ln_val - mean_ln_val) ** 2))
    rss_log = float(np.sum((ln_val - fitted_log) ** 2))
    loglog_r2 = 1.0 - rss_log / tss_log
    loglog_rmse = math.sqrt(rss_log / n)

    # Funnel check on log model
    sort_order_log = np.argsort(fitted_log, kind="stable")
    resid_log = ln_val - fitted_log
    resid_lower_log = resid_log[sort_order_log[:400]]
    resid_upper_log = resid_log[sort_order_log[400:]]
    resid_sd_lower_log = float(np.std(resid_lower_log, ddof=1))
    resid_sd_upper_log = float(np.std(resid_upper_log, ddof=1))
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # 1.7 Multiple regression by normal equation
    # Order: 1, ln(X4), X1, X2, X3, X5, X6, X7
    ones = np.ones(n)
    X_mat = np.column_stack([ones, ln_X4, X1, X2, X3, X5, X6, X7])
    Y_vec = ln_val

    # beta_hat = (X^T X)^(-1) X^T Y
    beta_hat = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ Y_vec)

    fitted_mr = X_mat @ beta_hat
    rss_mr = float(np.sum((Y_vec - fitted_mr) ** 2))
    mr_r2 = 1.0 - rss_mr / tss_log

    p_mr = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1)

    p_log = 1
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - p_log - 1)

    # Collect results
    results = [
        ("n", n),
        ("mean_value", mean_value),
        ("sd_value", sd_value),
        ("se_value", se_value),
        ("ci95_low", ci95_low),
        ("ci95_high", ci95_high),
        ("z_stat", z_stat),
        ("reject_h0", reject_h0),
        ("n_required", n_required),
        ("reg_slope", reg_slope),
        ("reg_intercept", reg_intercept),
        ("tss", tss),
        ("ess", ess),
        ("rss", rss),
        ("tss_equals_ess_plus_rss", tss_equals_ess_plus_rss),
        ("r2", r2),
        ("rmse", rmse),
        ("mae", mae),
        ("gd_J0", gd_costs[0]),
        ("gd_J1", gd_costs[1]),
        ("gd_J10", gd_costs[10]),
        ("gd_J100", gd_costs[100]),
        ("gd_J500", gd_costs[500]),
        ("gd_slope", gd_slope),
        ("gd_intercept", gd_intercept),
        ("gd_slope_gap", gd_slope_gap),
        ("resid_sd_lower_raw", resid_sd_lower_raw),
        ("resid_sd_upper_raw", resid_sd_upper_raw),
        ("resid_sd_ratio_raw", resid_sd_ratio_raw),
        ("loglog_intercept", loglog_intercept),
        ("loglog_slope", loglog_slope),
        ("loglog_r2", loglog_r2),
        ("loglog_rmse", loglog_rmse),
        ("resid_sd_lower_log", resid_sd_lower_log),
        ("resid_sd_upper_log", resid_sd_upper_log),
        ("resid_sd_ratio_log", resid_sd_ratio_log),
        ("mr_beta_0", beta_hat[0]),
        ("mr_beta_1", beta_hat[1]),
        ("mr_beta_2", beta_hat[2]),
        ("mr_beta_3", beta_hat[3]),
        ("mr_beta_4", beta_hat[4]),
        ("mr_beta_5", beta_hat[5]),
        ("mr_beta_6", beta_hat[6]),
        ("mr_beta_7", beta_hat[7]),
        ("mr_r2", mr_r2),
        ("mr_adj_r2", mr_adj_r2),
        ("loglog_adj_r2", loglog_adj_r2),
    ]

    write_results_file("part1_results.txt", results)
    return results


if __name__ == "__main__":
    run_part1()
