"""Part 3: Segmentation and dimensionality reduction."""
import math
import numpy as np
from common import load_data, write_results_file, write_csv_file


def run_kmeans(X_std, K):
    """
    K-means clustering from scratch with pinned rules:
    Starting centroids: 0, floor(800/K), ..., (K-1)*floor(800/K)
    Nearest centroid by squared Euclidean distance (ties to lower-numbered centroid).
    Centroid updated to mean of members (empty centroid stays).
    Stop when no row changes cluster.
    """
    n = len(X_std)
    init_indices = [i * (n // K) for i in range(K)]
    centroids = X_std[init_indices].copy()

    prev_assignments = np.full(n, -1, dtype=int)
    assignment_passes = 0

    while True:
        assignment_passes += 1
        # Squared Euclidean distance to each centroid: shape (n, K)
        # (x - c)^2 = sum(x^2) - 2 x.c + sum(c^2)
        dists = np.zeros((n, K), dtype=np.float64)
        for k in range(K):
            dists[:, k] = np.sum((X_std - centroids[k]) ** 2, axis=1)

        # Ties go to lower-numbered centroid: np.argmin returns first occurrence of minimum
        assignments = np.argmin(dists, axis=1)

        if np.array_equal(assignments, prev_assignments):
            break

        prev_assignments = assignments.copy()

        # Update centroids
        for k in range(K):
            members = X_std[assignments == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)

    # Calculate WCSS
    wcss = 0.0
    for k in range(K):
        members = X_std[assignments == k]
        if len(members) > 0:
            wcss += float(np.sum((members - centroids[k]) ** 2))

    return assignments, centroids, wcss, assignment_passes


def run_hierarchical(D_table, labels, linkage_rule):
    """
    Agglomerative hierarchical clustering from scratch.
    Linkage: 'single', 'complete', 'average'.
    At each step merge two clusters with smallest linkage distance (no ties occur).
    Returns list of merge records: (merge_no, cluster_a, cluster_b, height).
    """
    # Initialize clusters: each point is its own cluster
    # cluster represented by list of member indices
    clusters = [[i] for i in range(len(labels))]
    merges = []

    for merge_no in range(1, len(labels)):
        min_dist = float("inf")
        best_pair = None

        # Compare all pairs of active clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1 = clusters[i]
                c2 = clusters[j]

                # Distances between all pairs of members
                member_dists = [D_table[m1, m2] for m1 in c1 for m2 in c2]

                if linkage_rule == "single":
                    dist = min(member_dists)
                elif linkage_rule == "complete":
                    dist = max(member_dists)
                elif linkage_rule == "average":
                    dist = sum(member_dists) / len(member_dists)
                else:
                    raise ValueError(f"Unknown linkage: {linkage_rule}")

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j)

        i, j = best_pair
        c1 = clusters[i]
        c2 = clusters[j]

        # Format cluster names: member names in ascending order joined by '+'
        # cluster_a is the cluster whose smallest member number is the smaller one
        name_1 = "+".join(sorted([labels[m] for m in c1]))
        name_2 = "+".join(sorted([labels[m] for m in c2]))

        # Member number comparison (e.g. E1 < E2)
        min_member_1 = min(c1)
        min_member_2 = min(c2)

        if min_member_1 <= min_member_2:
            cluster_a_str = name_1
            cluster_b_str = name_2
        else:
            cluster_a_str = name_2
            cluster_b_str = name_1

        merges.append((merge_no, cluster_a_str, cluster_b_str, min_dist))

        # Merge c1 and c2
        new_cluster = c1 + c2
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(new_cluster)

    return merges


def run_part3():
    data = load_data()
    X = data["X"]  # X1..X7
    y = data["y"]  # outcome
    value = data["value"]

    # Continuous predictors X1 to X5
    X_cont = X[:, 0:5]
    n = len(X_cont)

    # 3.1 Standardise X1 to X5 over all 800 rows (sample SD, ddof=1)
    mean_cont = np.mean(X_cont, axis=0)
    sd_cont = np.std(X_cont, axis=0, ddof=1)
    X_std = (X_cont - mean_cont) / sd_cont

    # Run K-means for K = 2, 3, 4, 5, 6
    kmeans_results = {}
    for K in [2, 3, 4, 5, 6]:
        assigns, centroids, wcss, passes = run_kmeans(X_std, K)
        kmeans_results[K] = {
            "assignments": assigns,
            "centroids": centroids,
            "wcss": wcss,
            "passes": passes
        }

    wcss_k2 = kmeans_results[2]["wcss"]
    wcss_k3 = kmeans_results[3]["wcss"]
    wcss_k4 = kmeans_results[4]["wcss"]
    wcss_k5 = kmeans_results[5]["wcss"]
    wcss_k6 = kmeans_results[6]["wcss"]

    # 3.2 Choose K with the elbow
    # Smallest K in {3, 4, 5} for which:
    # WCSS(K) - WCSS(K + 1) < 0.5 * (WCSS(K - 1) - WCSS(K))
    wcss_map = {k: kmeans_results[k]["wcss"] for k in [2, 3, 4, 5, 6]}
    elbow_k = None
    for K in [3, 4, 5]:
        drop_next = wcss_map[K] - wcss_map[K + 1]
        drop_prev = wcss_map[K - 1] - wcss_map[K]
        if drop_next < 0.5 * drop_prev:
            elbow_k = K
            break

    # 3.3 Profile the K = 3 segments
    # Order clusters by ascending mean of X3 (in original units) as A, B, C
    assigns_3 = kmeans_results[3]["assignments"]
    cluster_means_X3 = []
    for c in range(3):
        mask = (assigns_3 == c)
        mean_x3 = float(np.mean(X_cont[mask, 2]))  # X3 is column index 2
        cluster_means_X3.append((c, mean_x3))

    # Sort clusters by mean of X3 ascending
    sorted_clusters = sorted(cluster_means_X3, key=lambda x: x[1])
    seg_names = ["A", "B", "C"]
    cluster_to_seg = {sorted_clusters[i][0]: seg_names[i] for i in range(3)}

    # Vector of segment names for each record
    record_segments = np.array([cluster_to_seg[c] for c in assigns_3])

    segment_rows = []
    segment_sizes = {}
    for seg_name in ["A", "B", "C"]:
        mask = (record_segments == seg_name)
        size = int(np.sum(mask))
        segment_sizes[seg_name] = size

        means_X = np.mean(X_cont[mask], axis=0)
        outcome_rate = float(np.mean(y[mask]))
        mean_val = float(np.mean(value[mask]))

        segment_rows.append([
            seg_name,
            size,
            means_X[0],
            means_X[1],
            means_X[2],
            means_X[3],
            means_X[4],
            outcome_rate,
            mean_val
        ])

    write_csv_file(
        "part3_segments.csv",
        ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
        segment_rows
    )

    # 3.4 Hierarchical clustering on supplied table
    labels = ["E1", "E2", "E3", "E4", "E5", "E6"]
    D_table = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ], dtype=np.float64)

    hier_csv_rows = []
    cut_clusters = {}

    for rule in ["single", "complete", "average"]:
        merges = run_hierarchical(D_table, labels, rule)
        for m in merges:
            hier_csv_rows.append([rule, m[0], m[1], m[2], m[3]])

        # Cut dendrogram:
        # Four gaps between consecutive merge heights:
        # gap 0 = m2 - m1, gap 1 = m3 - m2, gap 2 = m4 - m3, gap 3 = m5 - m4
        heights = [m[3] for m in merges]
        gaps = [heights[i + 1] - heights[i] for i in range(4)]
        # Cut lies in the largest gap
        # If largest gap is at index i (0..3), that means between merge (i+1) and merge (i+2).
        # Number of merges below the cut is (i + 1).
        # Number of clusters is 6 - (i + 1).
        largest_gap_idx = int(np.argmax(gaps))
        n_merges_below_cut = largest_gap_idx + 1
        clusters_at_cut = 6 - n_merges_below_cut
        cut_clusters[rule] = clusters_at_cut

    write_csv_file(
        "part3_hierarchical.csv",
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hier_csv_rows
    )

    # 3.5 PCA by seven steps
    # (1) mean vector: all zeros
    # (2) centre data: already centred
    # (3) Covariance matrix S = (1/(n-1)) X^T X
    S = (1.0 / (n - 1)) * (X_std.T @ X_std)

    # (4) Eigenvalues and eigenvectors using numpy.linalg.eigh
    eigenvalues_raw, eigenvectors_raw = np.linalg.eigh(S)

    # (5) Sort by eigenvalue, largest first
    sort_idx = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues = eigenvalues_raw[sort_idx]
    eigenvectors = eigenvectors_raw[:, sort_idx]

    # Sign convention: flip each eigenvector so that its entry of largest absolute value is positive
    for j in range(eigenvectors.shape[1]):
        max_abs_idx = int(np.argmax(np.abs(eigenvectors[:, j])))
        if eigenvectors[max_abs_idx, j] < 0:
            eigenvectors[:, j] = -eigenvectors[:, j]

    eigenvalue_sum = float(np.sum(eigenvalues))
    explained_ratios = eigenvalues / eigenvalue_sum
    cumulative_ratios = np.cumsum(explained_ratios)

    # Smallest number of components with cumulative ratio >= 0.8
    components_80 = int(np.argmax(cumulative_ratios >= 0.8) + 1)

    # Top two eigenvectors (W)
    W2 = eigenvectors[:, :2]
    # Project Z = X W
    Z = X_std @ W2

    pc1_loadings = eigenvectors[:, 0]
    pc2_loadings = eigenvectors[:, 1]

    # Projections of first three data rows (indices 0, 1, 2)
    row1_pc1, row1_pc2 = Z[0, 0], Z[0, 1]
    row2_pc1, row2_pc2 = Z[1, 0], Z[1, 1]
    row3_pc1, row3_pc2 = Z[2, 0], Z[2, 1]

    # Mean PC1 segment scores
    mean_pc1_A = float(np.mean(Z[record_segments == "A", 0]))
    mean_pc1_B = float(np.mean(Z[record_segments == "B", 0]))
    mean_pc1_C = float(np.mean(Z[record_segments == "C", 0]))

    results = [
        ("wcss_k2", wcss_k2),
        ("wcss_k3", wcss_k3),
        ("wcss_k4", wcss_k4),
        ("wcss_k5", wcss_k5),
        ("wcss_k6", wcss_k6),
        ("elbow_k", elbow_k),
        ("size_segment_A", segment_sizes["A"]),
        ("size_segment_B", segment_sizes["B"]),
        ("size_segment_C", segment_sizes["C"]),
        ("single_clusters_at_cut", cut_clusters["single"]),
        ("complete_clusters_at_cut", cut_clusters["complete"]),
        ("average_clusters_at_cut", cut_clusters["average"]),
        ("eigenvalue_sum", eigenvalue_sum),
        ("eigenvalue_1", eigenvalues[0]),
        ("eigenvalue_2", eigenvalues[1]),
        ("eigenvalue_3", eigenvalues[2]),
        ("eigenvalue_4", eigenvalues[3]),
        ("eigenvalue_5", eigenvalues[4]),
        ("explained_ratio_1", explained_ratios[0]),
        ("explained_ratio_2", explained_ratios[1]),
        ("explained_ratio_3", explained_ratios[2]),
        ("explained_ratio_4", explained_ratios[3]),
        ("explained_ratio_5", explained_ratios[4]),
        ("cumulative_ratio_1", cumulative_ratios[0]),
        ("cumulative_ratio_2", cumulative_ratios[1]),
        ("cumulative_ratio_3", cumulative_ratios[2]),
        ("cumulative_ratio_4", cumulative_ratios[3]),
        ("cumulative_ratio_5", cumulative_ratios[4]),
        ("components_for_80pct", components_80),
        ("pc1_loading_X1", pc1_loadings[0]),
        ("pc1_loading_X2", pc1_loadings[1]),
        ("pc1_loading_X3", pc1_loadings[2]),
        ("pc1_loading_X4", pc1_loadings[3]),
        ("pc1_loading_X5", pc1_loadings[4]),
        ("pc2_loading_X1", pc2_loadings[0]),
        ("pc2_loading_X2", pc2_loadings[1]),
        ("pc2_loading_X3", pc2_loadings[2]),
        ("pc2_loading_X4", pc2_loadings[3]),
        ("pc2_loading_X5", pc2_loadings[4]),
        ("row1_pc1", row1_pc1),
        ("row1_pc2", row1_pc2),
        ("row2_pc1", row2_pc1),
        ("row2_pc2", row2_pc2),
        ("row3_pc1", row3_pc1),
        ("row3_pc2", row3_pc2),
        ("mean_pc1_segment_A", mean_pc1_A),
        ("mean_pc1_segment_B", mean_pc1_B),
        ("mean_pc1_segment_C", mean_pc1_C),
    ]

    write_results_file("part3_results.txt", results)
    return results


if __name__ == "__main__":
    run_part3()
