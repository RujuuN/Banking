"""
Self-verification script to check that the entire pipeline is working properly.
Run with: python verify_all.py
"""
import csv
import hashlib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_FILE = os.path.join(BASE_DIR, "data", "records.csv")


def run_checks():
    print("=" * 60)
    print("  PhonePe Banking Capstone — Pipeline Verification")
    print("=" * 60)
    passed = 0
    total = 0

    def record_check(description, condition, details=""):
        nonlocal passed, total
        total += 1
        status = "[PASS]" if condition else "[FAIL]"
        if condition:
            passed += 1
            print(f"{status} {description}")
        else:
            print(f"{status} {description}")
            if details:
                print(f"       -> Details: {details}")

    # 1. Data Check
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        pos_count = sum(1 for r in rows[1:] if r[8] == "1")
        data_valid = (len(rows) == 801 and len(rows[0]) == 10 and pos_count == 170 and digest == "e85e7605b784f4142844f79ad2dd0b8e")
        record_check("Dataset data/records.csv (800 rows, 170 defaults, MD5 verified)", data_valid, f"Rows: {len(rows)-1}, Defaults: {pos_count}, MD5: {digest}")
    else:
        record_check("Dataset data/records.csv exists", False, "File missing")

    # 2. Check Results Files Exist
    results_map = {}
    for part in [1, 2, 3, 4]:
        fname = f"part{part}_results.txt"
        fpath = os.path.join(OUTPUTS_DIR, fname)
        exists = os.path.exists(fpath)
        record_check(f"Results file outputs/{fname} exists", exists)
        if exists:
            kv = {}
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if " = " in line:
                        k, v = line.strip().split(" = ", 1)
                        kv[k] = v
            results_map[part] = kv

    # 3. Check Part 1 Checkpoints
    p1 = results_map.get(1, {})
    record_check("Part 1: n = 800", p1.get("n") == "800", f"Got {p1.get('n')}")
    record_check("Part 1: mean_value = 8.9382", p1.get("mean_value") == "8.9382", f"Got {p1.get('mean_value')}")
    record_check("Part 1: gd_slope_gap < 0.001", float(p1.get("gd_slope_gap", 999)) < 0.001, f"Got {p1.get('gd_slope_gap')}")
    record_check("Part 1: tss_equals_ess_plus_rss = yes", p1.get("tss_equals_ess_plus_rss") == "yes", f"Got {p1.get('tss_equals_ess_plus_rss')}")

    # 4. Check Part 2 Checkpoints
    p2 = results_map.get(2, {})
    record_check("Part 2: train_rows = 560", p2.get("train_rows") == "560", f"Got {p2.get('train_rows')}")
    record_check("Part 2: train_positives = 119", p2.get("train_positives") == "119", f"Got {p2.get('train_positives')}")
    record_check("Part 2: test_rows = 240", p2.get("test_rows") == "240", f"Got {p2.get('test_rows')}")
    record_check("Part 2: test_positives = 51", p2.get("test_positives") == "51", f"Got {p2.get('test_positives')}")
    record_check("Part 2: loss_epoch_0 = 0.6931", p2.get("loss_epoch_0") == "0.6931", f"Got {p2.get('loss_epoch_0')}")
    cost_tau_val = float(p2.get("cost_tau", 0))
    is_mult_005 = abs(cost_tau_val / 0.05 - round(cost_tau_val / 0.05)) < 1e-5
    record_check("Part 2: cost_tau is multiple of 0.05", is_mult_005, f"Got {cost_tau_val}")

    # 5. Check Part 3 Checkpoints
    p3 = results_map.get(3, {})
    record_check("Part 3: wcss_k2 = 2262.6753", p3.get("wcss_k2") == "2262.6753", f"Got {p3.get('wcss_k2')}")
    record_check("Part 3: eigenvalue_sum = 5.0000", p3.get("eigenvalue_sum") == "5.0000", f"Got {p3.get('eigenvalue_sum')}")
    total_seg = int(p3.get("size_segment_A", 0)) + int(p3.get("size_segment_B", 0)) + int(p3.get("size_segment_C", 0))
    record_check("Part 3: Segment sizes sum to 800", total_seg == 800, f"Got {total_seg}")

    # 6. Check Part 4 Checkpoints
    p4 = results_map.get(4, {})
    record_check("Part 4: root_entropy = 0.7462", p4.get("root_entropy") == "0.7462", f"Got {p4.get('root_entropy')}")
    record_check("Part 4: round1_error = 0.2500", p4.get("round1_error") == "0.2500", f"Got {p4.get('round1_error')}")

    # 7. Check All CSV Files
    csv_specs = {
        "part2_thresholds.csv": ("tau,tp,fp,tn,fn,tpr,fpr", 9),
        "part2_knn_cv.csv": ("K,mean_cv_f1", 11),
        "part3_segments.csv": ("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value", 3),
        "part3_hierarchical.csv": ("linkage,merge_no,cluster_a,cluster_b,height", 15),
        "part4_splits.csv": ("split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain", 4),
        "part4_voting.csv": ("model,tp,fp,tn,fn,precision,recall,f1", 5),
        "part4_adaboost.csv": ("round,point,weight_before,weight_unnormalised,weight_after", 16),
        "part4_architecture.csv": ("layer,parameters", 7),
    }

    for fname, (exp_header, exp_rows) in csv_specs.items():
        fpath = os.path.join(OUTPUTS_DIR, fname)
        if not os.path.exists(fpath):
            record_check(f"CSV outputs/{fname} exists and format valid", False, "File missing")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        act_header = ",".join(reader[0]) if reader else ""
        act_rows = len(reader) - 1
        valid = (act_header == exp_header and act_rows == exp_rows)
        record_check(f"CSV outputs/{fname} ({exp_rows} rows, exact header)", valid, f"Header: '{act_header}', Rows: {act_rows}")

    # 8. Check README formatting
    readme_path = os.path.join(BASE_DIR, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            rcontent = f.read()
        lines = rcontent.splitlines()
        first_line_ok = (len(lines) > 0 and lines[0].strip() == "Domain: Banking")
        record_check("README.md first line is 'Domain: Banking'", first_line_ok, f"Got '{lines[0] if lines else ''}'")

        memo_start = rcontent.find("## Model-choice memo")
        memo_end = rcontent.find("## Limitations")
        if memo_start != -1 and memo_end != -1:
            memo_text = rcontent[memo_start:memo_end].replace("## Model-choice memo", "").strip()
            word_count = len(memo_text.split())
            record_check(f"Model-choice memo length <= 300 words ({word_count} words)", word_count <= 300, f"Got {word_count} words")
        else:
            record_check("Model-choice memo present in README", False, "Heading missing")

    print("=" * 60)
    print(f"  Summary: {passed} / {total} checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()
