Domain: Banking

# PhonePe Banking Decision Intelligence — Capstone Project

This repository provides an end-to-end analytics and machine learning pipeline for PhonePe's digital lending-risk operations. It investigates borrower monthly spend assumptions, builds calibrated default classification models, segments the loan portfolio, and evaluates deep neural network architectures for document OCR and remark processing.

---

## How to run

The entire pipeline runs offline with standard Python (3.10+) and NumPy. To reproduce all results and regenerate every committed artifact in `outputs/`, execute the following commands from a clean clone in the repository root:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic banking records (verifies MD5 checksum)
python generate_data.py

# 3. Run Part 1: Estimation, inference, OLS, gradient descent, and multiple regression
python part1.py

# 4. Run Part 2: Stratified split, logistic regression, KNN CV, and cost optimization
python part2.py

# 5. Run Part 3: K-means clustering, elbow selection, hierarchical linkage, and PCA
python part3.py

# 6. Run Part 4: Decision stump, voting ensembles, AdaBoost, and neural network sizing
python part4.py
```

---

## Part 1 findings

- **R1.1 Planning Claim Hypothesis Test**: Testing the budget assumption H0: mu = 8.0 against H1: mu != 8.0 yielded a sample mean of 8.9382 thousand INR (INR 8,938.20), a standard error of 0.3243, and a test statistic of `z_stat = 2.8930`. Because |z_stat| = 2.8930 > 1.96, we reject H0 at the alpha = 0.05 significance level (`reject_h0 = yes`). A Type I error in this context would occur if PhonePe rejected the INR 8,000 monthly spend planning assumption and needlessly reallocated lending liquidity when the true population mean spend was in fact exactly INR 8,000. The observed difference is 0.9382 column units (INR 938.20 per borrower per month, or an 11.7% increase above budget); this gap matters substantially in practice because across a portfolio of tens of thousands of active credit lines, an unbudgeted spend of nearly INR 940 per user creates severe liquidity shortfalls for funding partners.
- **R1.2 Sample Sizing**: To estimate the mean spend within a margin of error of INR 500 (E = 0.5 thousand INR) at 95% confidence given the observed standard deviation of 9.1723, the required sample size is `n_required = 1293`. Because PhonePe currently has only 800 borrower rows available, the sample is insufficient for the desired precision; the planning team must either collect approximately 500 additional borrower records or accept a wider confidence margin of error (+/- 1.96 * 0.3243 approx +/- INR 635.60).
- **R1.3 Gradient Descent vs. Closed Form**: The slope gap between closed-form ordinary least squares and batch gradient descent is `gd_slope_gap = 0.0000`. Gradient descent reached the exact analytical line because standardized linear regression minimizes a strictly convex quadratic cost function with a single global optimum, ensuring that gradient descent with learning rate eta = 0.1 across 500 epochs converges precisely to the normal equation solution.
- **R1.4 Heteroscedasticity Funnel and Log Transform**: The raw regression model exhibits a severe funnel effect, with residual standard deviation expanding from 2.3155 in the lower half of fitted values to 8.6891 in the upper half (`resid_sd_ratio_raw = 3.7525`). Regressing ln(value) on ln(X4) eliminates this heteroscedasticity, shrinking the spread ratio to `resid_sd_ratio_log = 1.0396` and confirming that multiplicative spend scaling is effectively stabilized by the logarithmic remedy.
- **R1.5 Multiple Regression Predictors**: In the multiple regression of ln(value) on all features, the predictors with an absolute coefficient above 0.05 are ln(X4) (`avg_monthly_inflow_inr_thousands`, beta_1 = 1.1576), X2 (`failed_txn_count_90d`, beta_3 = 0.0940), X3 (`credit_utilisation_ratio`, beta_4 = 0.1067), and X6 (`irregular_salary_flag`, beta_6 = 0.1518). Adjusted R-squared rises from `loglog_adj_r2 = 0.6561` in the bivariate log-log model to `mr_adj_r2 = 0.7403` in the multiple regression model, proving that incorporating tenure, payment frictions, utilization, and salary stability explains significant incremental variance even after penalizing for extra degrees of freedom.

---

## Part 2 findings

- **R2.1 Learning Rate Comparison**: Ordering the training losses at epoch 200 from lowest to highest gives eta = 1.0 (`loss_epoch_200_eta_1.0 = 0.3703`), eta = 0.1 (`loss_epoch_200_eta_0.1 = 0.3776`), and eta = 0.01 (`loss_epoch_200_eta_0.01 = 0.5091`). This demonstrates that a learning rate that is too small (eta = 0.01) converges sluggishly and remains underfitted, whereas a large learning rate (eta = 1.0) accelerates along the smooth convex logistic surface to reach near-optimal loss without overshooting or divergence.
- **R2.2 Top Predictors and Odds Ratios**: The three predictors with the largest absolute model weights are X3 (`credit_utilisation_ratio`, w = 0.9013), X1 (`account_age_months`, w = -0.5554), and X2 (`failed_txn_count_90d`, w = 0.5505). The largest odds ratio is `odds_ratio_X3 = 2.4629`, meaning that each one-standard-deviation increase in a borrower's credit line utilisation ratio multiplies the odds of loan default by approximately 2.46 (a 146.3% increase in default odds), holding all other predictors constant.
- **R2.3 Accuracy Inadequacy and Threshold Shift**: The naive benchmark predicting zero default for every record yields `blind_accuracy = 0.7875`, which is nearly identical to the logistic model's accuracy of `tau_0.5_accuracy = 0.8042`. Accuracy is a severely misleading metric in imbalanced lending books because a model predicting zero defaults achieves high accuracy while catching zero credit losses. Lowering the decision threshold from tau = 0.5 to tau = 0.3 increased recall from 0.4706 to 0.7451 (+27.45 percentage points, catching 38 of 51 defaults) while precision decreased moderately from 0.5455 to 0.4634.
- **R2.4 Logistic vs. K-Nearest Neighbours**: Comparing the logistic regression model at tau = 0.3 with the 5-nearest neighbours model on the held-out test set:
  - Logistic (tau = 0.3): `f1 = 0.5714`, `recall = 0.7451`
  - KNN (K = 5): `knn_f1 = 0.3696`, `knn_recall = 0.3333`
  The logistic model decisively outperforms KNN, capturing more than double the proportion of defaulting borrowers.
- **R2.5 Cost Optimization and Deployment**: Based on PhonePe's operational cost structure (INR 50 per false alarm call, INR 400 per missed default recovery), we recommend deploying the logistic regression model at the cost-optimal alert threshold of `cost_tau = 0.1500`. On the test set, this policy achieves `cost_test_at_tau = INR 6050`, cutting total collection costs almost in half compared to `cost_test_at_0.5 = INR 11800` and saving INR 14,350 compared to flagging nobody (`cost_test_flag_nobody = INR 20400`). Tuning the decision threshold exclusively on training records prevented data leakage and ensures unbiased cost estimation on unseen cohorts.

---

## Part 3 findings

- **R3.1 K-Means Elbow Analysis**: The successive WCSS drops across cluster counts are:
  - WCSS(2) - WCSS(3) = 2262.6753 - 1602.8471 = 659.8282
  - WCSS(3) - WCSS(4) = 1602.8471 - 1494.7732 = 108.0739
  - WCSS(4) - WCSS(5) = 1494.7732 - 1323.7157 = 171.0575
  - WCSS(5) - WCSS(6) = 1323.7157 - 1278.1691 = 45.5466
  The elbow criterion formally selects `elbow_k = 3` because the drop from K = 3 to K = 4 (108.0739) is less than half the preceding drop from K = 2 to K = 3 (0.5 * 659.8282 = 329.9141). Although the drop from K = 4 to K = 5 slightly rebounds because pinned deterministic starting rows yield a reproducible local partition rather than a global optimum, K = 3 represents the primary structural elbow.
- **R3.2 Borrower Segment Profiles and Actions**:
  - **Segment A ("Established Prime Transactors", Size: 322)**: Characterized by mature tenure (mean X1 = 47.50 months), minimal friction (mean X2 = 1.95), low credit utilization (mean X3 = 0.2250), strong repayment scores (mean X5 = 62.16), and very low default risk (`outcome_rate = 0.0186` or 1.86%). *Action*: Extend pre-approved credit limits and premium merchant rewards to stimulate spend.
  - **Segment B ("High-Volume Power Spenders", Size: 255)**: Displays moderate tenure (28.61 months), highest monthly inflow (INR 28,719), highest spend (INR 17,125), and elevated friction (mean X2 = 6.07). Default rate is moderate (`outcome_rate = 0.2353` or 23.53%). *Action*: Deploy automated payment retry alerts and incentivize autopay mandate registration.
  - **Segment C ("New High-Utilisation Distress Borrowers", Size: 223)**: Marked by short account tenure (14.39 months), severe credit line strain (mean X3 = 0.7118), and depressed repayment scores (mean X5 = 38.52). Segment C exhibits the highest delinquency rate by far (`outcome_rate = 0.4664` or 46.64%). *Action*: Cap borrowing limits, mandate weekly repayment installments, and schedule early collections nudges 5 days prior to due dates.
- **R3.3 Hierarchical Clustering Cut Disagreement**: Cutting each dendrogram at its largest merge height gap produces `single_clusters_at_cut = 4` clusters for single linkage (`{E2, E3, E6}`, `{E1}`, `{E4}`, `{E5}`) versus `complete_clusters_at_cut = 2` clusters for complete linkage (`{E1, E4, E5}` and `{E2, E3, E6}`). Single linkage merges based on minimum pairwise distances, causing chaining where intermediate close cohorts bridge early while isolated points remain unmerged singletons until late in the hierarchy. In contrast, complete linkage evaluates cluster diameter (maximum pairwise distance), forcing compact, fully connected cohorts to aggregate into two balanced macro-clusters before making the large final merge across disparate groups at height 7.9.
- **R3.4 Principal Component Decomposition**: The first principal component is primarily driven in the positive direction by account tenure X1 (loading: +0.5317) and repayment score X5 (loading: +0.4418), and in the negative direction by credit utilisation ratio X3 (loading: -0.5187) and failed transaction counts X2 (loading: -0.4100). Exactly `components_for_80pct = 3` components are required to capture at least 80% of total variance (PC1: 52.27%, PC2: 23.66%, PC3: 10.23%; cumulative: 86.16%). Segment mean PC1 scores (`mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = -0.7831`, `mean_pc1_segment_C = -1.6709`) demonstrate that PC1 reflects underlying credit health, cleanly segregating prime low-risk borrowers (positive space) from distressed cohorts (negative space).

---

## Part 4 findings

- **R4.1 Decision Stump Selection and Tree Regularization**: Split S4 (X1 <= 24.0 months) is the purest split, yielding the highest information gain (`info_gain = 0.1252`) and Gini gain (`gini_gain = 0.0606`) because tenure under two years cleanly isolates early-stage borrowers with high default rates (45.74%, 86/188) from seasoned accounts (8.87%, 33/372). Growing a decision tree until every leaf is pure causes severe overfitting because the tree memorizes idiosyncratic sample noise and statistical flukes rather than true population patterns. Setting a maximum depth limit (`max_depth`) or minimum samples per leaf (`min_samples_leaf`) halts recursive splitting before spurious leaves form.
- **R4.2 Model Voting Comparison**: Evaluating models on the 240 test records confirms that standalone Logistic Regression should be shipped:
  - Logistic (tau = 0.3): `f1 = 0.5714`, `recall = 0.7451`
  - KNN (K = 5): `knn_f1 = 0.3696`, `knn_recall = 0.3333`
  - Decision Stump (S4): `stump_f1 = 0.5532`, `stump_recall = 0.7647`
  - Hard Voting: `hard_vote_f1 = 0.5344`, `hard_vote_recall = 0.6863`
  - Soft Voting: `soft_vote_f1 = 0.5507`, `soft_vote_recall = 0.7451`
  Combining models did not improve performance; both hard and soft ensembles underperformed standalone logistic regression because the weak KNN classifier diluted ensemble predictions.
- **R4.3 AdaBoost Mechanics**: In round 1, weak classifier h1 misclassifies Points 3 and 7, resulting in weighted error eps1 = 0.2500 and voting weight alpha1 = 0.5493. Following weight updating and normalization, each misclassified point receives a weight of 0.2500, yielding a combined weight of `0.5000`. Round 2 trusts rule h2 substantially more than h1 because h2 achieves a lower weighted training error (eps2 = 0.1667 vs. eps1 = 0.2500), earning a higher voting weight of alpha2 = 0.8047 (compared to alpha1 = 0.5493) and resulting in an ensemble accuracy of 0.7500.
- **R4.4 Deep Learning Architecture Sizing**:
  
  **Model A Option 2 (CNN) Layer Breakdown:**
  | Layer | Output Shape (H x W x D) | Parameter Count |
  |---|---|---|
  | Input | 28 x 28 x 1 | 0 |
  | `cnn_conv1` (Conv 5x5, 6 filters, pad 2, stride 1) | 28 x 28 x 6 | 156 |
  | ReLU + MaxPool (2x2, stride 2) | 14 x 14 x 6 | 0 |
  | `cnn_conv2` (Conv 5x5, 16 filters, pad 0, stride 1) | 10 x 10 x 16 | 2,416 |
  | ReLU + MaxPool (2x2, stride 2) | 5 x 5 x 16 | 0 |
  | Flatten | 400 | 0 |
  | `cnn_dense1` (Dense 32, ReLU) | 32 | 12,832 |
  | `cnn_dense2` (Dense 10 Output) | 10 | 330 |
  | **Total CNN Parameters** | — | **15,734** |

  - **Convolutional Scaling**: For an input image of 128 x 128 x 3, a fully connected layer with 1,024 neurons requires `fc_layer_params = 50332672` parameters, whereas a convolutional layer with 64 filters of size 3 x 3 requires only `conv_layer_params = 1792` parameters due to local receptive fields and weight sharing.
  - **Model B (RNN) Sizing**: The recurrent network parameter matrices are W_xh in R^(128 x 64) (8,192), W_hh in R^(128 x 128) (16,384), b_h in R^128 (128), W_hy in R^(16 x 128) (2,048), and b_y in R^16 (16), totaling `rnn_total = 26768` parameters. Total parameters do not grow with longer text sequences because recurrent weights are reused across every time step.
  - **Activations and Loss**: For 10-class digit recognition, we select ReLU activation for hidden layers and Softmax activation paired with Categorical Cross-Entropy loss for output classification.
  - **Vanishing Gradients and Residual Connections**: In deep plain neural networks, multiplying repeated fractional derivatives during backpropagation causes error gradients to diminish exponentially as they propagate backward, preventing early layers from updating their weights. Residual skip connections bypass intermediate layers by adding identity mappings (F(x) + x), creating an unobstructed gradient highway that enables effective backpropagation throughout deep architectures.

---

## Model-choice memo

**To:** Head of Lending-Risk Analytics, PhonePe  
**From:** Decision Intelligence Team  
**Subject:** Recommended Default Prediction Model for Production Deployment  

We recommend deploying the standalone Logistic Regression classifier with an alert threshold of tau = 0.1500 for operational collections nudges.

Evaluating models under PhonePe's economic loss function (INR 50 per false alert, INR 400 per missed default), the logistic model at tau = 0.1500 achieves an expected test cost of INR 6,050. This cuts total collection losses nearly in half compared to the standard tau = 0.5000 benchmark cost of INR 11,800, and saves INR 14,350 relative to the unguided baseline of flagging nobody (cost: INR 20,400). At this threshold, the model catches 48 of 51 test defaults (recall: 0.9412), ensuring high-risk borrowers receive pre-due-date reminders before entering expensive delinquency.

Ensemble methods failed to surpass the standalone logistic baseline: hard voting achieved an F1 score of 0.5344 and soft voting reached 0.5507, both below the logistic model's F1 of 0.5714 at tau = 0.3000, primarily because the K-nearest neighbours model performed poorly (F1: 0.3696, recall: 0.3333). Logistic regression also provides transparent odds ratios (e.g., credit utilisation ratio multiplies default odds by 2.4629 per standard deviation) and sub-millisecond inference suitable for real-time UPI credit workflows.

A primary risk is that findings derive from a single synthetic dataset containing only 51 test defaults without temporal out-of-time validation, making performance sensitive to distribution shifts and credit-cycle turns.

Neither Model A (cheque digit OCR) nor Model B (payment remark classification) should be built before this structured model is operational in production, as digitizing peripheral documents cannot match the immediate balance-sheet impact and fraud mitigation of core default prevention.

---

## Limitations

1. **Synthetic Data Limitations**: The data was generated using fixed parametric Box-Muller distributions and static logistic equations. Real-world UPI lending environments feature macroeconomic regime changes, inflation cycles, seasonal spending surges (e.g., Diwali), and strategic fraud attempts that are absent from this synthetic book.
2. **Single Static Train-Test Partition**: Models were trained and evaluated on a single 70/30 stratified split rather than a rolling out-of-time validation window. In real banking operations, credit models must demonstrate stability across longitudinal cohorts to guard against feature drift and changing borrower demographics.
3. **Sparse Positive Test Events**: The test partition contains only 51 default events. With such a small sample of positives, a difference of just 1 or 2 misclassified records introduces swings of nearly 2 to 4 percentage points in recall and precision, indicating that wider operational confidence intervals must be monitored post-deployment.
