# grader.py
import numpy as np
import pandas as pd
from collections import defaultdict
import importlib.util, sys
import traceback

# --- synthetic dataset generator ---
def make_synthetic_df(n_groups=200, avg_group_size=5, n_classes=3, imbalance=0.15, random_state=None):
    rng = np.random.RandomState(random_state)
    # prepare groups
    group_sizes = rng.poisson(lam=avg_group_size, size=n_groups) + 1
    rows = []
    for gid, size in enumerate(group_sizes):
        # each group gets a "group-level" class tendency to make stratification nontrivial
        class_bias = rng.normal(loc=0,scale=1.2,size=n_classes)
        class_probs = np.exp(class_bias - class_bias.max())
        class_probs /= class_probs.sum()
        for i in range(size):
            label = rng.choice(np.arange(n_classes), p=class_probs)
            rows.append({'group': f'g{gid}', 'label': int(label)})
    df = pd.DataFrame(rows)
    # shuffle
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)
    # give a non-default index
    df.index = rng.permutation(np.arange(10000, 10000 + len(df)))
    return df

# --- helper checks ---
def no_group_leakage(df, assignments, group_col):
    """
    assignments: either list of (train_idx, val_idx) pairs or df with cv_fold column.
    """
    if isinstance(assignments, list):
        for train_idx, val_idx in assignments:
            train_groups = set(df.loc[train_idx, group_col].unique())
            val_groups = set(df.loc[val_idx, group_col].unique())
            if train_groups & val_groups:
                return False, "group_leak"
        return True, ""
    else:
        # df expected with cv_fold column
        if 'cv_fold' not in assignments.columns:
            return False, "no_cv_fold"
        for fold in sorted(assignments['cv_fold'].unique()):
            val_idx = assignments[assignments['cv_fold'] == fold].index
            train_idx = assignments[assignments['cv_fold'] != fold].index
            train_groups = set(df.loc[train_idx, group_col].unique())
            val_groups = set(df.loc[val_idx, group_col].unique())
            if train_groups & val_groups:
                return False, "group_leak"
        return True, ""

def stratification_ok(df, assignments, label_col, max_tol=0.132, mean_tol=0.065, min_label_fold=1):
    """Dual-threshold stratification check.

    Conditions per fold:
      1) Max per-class relative freq deviation <= max_tol
      2) Mean per-class relative freq deviation <= mean_tol
      3) Each globally present label appears at least min_label_fold times.
    """
    overall = df[label_col].value_counts(normalize=True).sort_index()
    labels = overall.index.tolist()
    if isinstance(assignments, list):
        folds_vals = []
        fold_indices = []
        for train_idx, val_idx in assignments:
            val_series = df.loc[val_idx, label_col].value_counts(normalize=True).reindex(labels, fill_value=0)
            folds_vals.append(val_series)
            fold_indices.append(val_idx)
    else:
        folds_vals = []
        fold_indices = []
        for fold in sorted(assignments['cv_fold'].unique()):
            val_idx = assignments[assignments['cv_fold'] == fold].index
            val_series = assignments.loc[val_idx, label_col].value_counts(normalize=True).reindex(labels, fill_value=0)
            folds_vals.append(val_series)
            fold_indices.append(val_idx)

    total_rows = len(df)
    ideal_size = total_rows / len(folds_vals) if folds_vals else 0
    for f_idx, val_dist in enumerate(folds_vals):
        diffs = (val_dist - overall).abs()
        max_dev = diffs.max()
        mean_dev = diffs.mean()
        if max_dev > max_tol:
            return False, f"strat_max_fail_fold_{f_idx}"
        if mean_dev > mean_tol:
            return False, f"strat_mean_fail_fold_{f_idx}"
    # fold size imbalance (soft advisory only): compute relative deviation but do not fail.
    size_rel_dev = abs(len(fold_indices[f_idx]) - ideal_size) / (ideal_size + 1e-9)
    # label presence
    for f_idx, val_idx in enumerate(fold_indices):
        counts = df.loc[val_idx, label_col].value_counts()
        for lbl in labels:
            if overall[lbl] > 0 and counts.get(lbl, 0) < min_label_fold:
                return False, f"label_missing_fold_{f_idx}_{lbl}"
        # missing class set (collect which labels absent entirely in this fold)
        absent = [lbl for lbl in labels if counts.get(lbl,0)==0 and overall[lbl]>0]
        if absent:
            return False, f"missing_classes_fold_{f_idx}_" + ",".join(map(str, absent))
        # if size issue existed for any fold but no other failure triggered, we do not fail; optionally could attach info.
    return True, ""

# --- grader runner ---
def grade_submission(submission_module, num_trials=10, n_splits=4):
    """
    submission_module: Python module object with stratified_group_k_fold function.
    Run across several synthetic datasets and seeds. Return pass rate and details.
    """
    results = []
    seeds = list(range(1001, 1001 + num_trials))
    for seed in seeds:
        df = make_synthetic_df(n_groups=150, avg_group_size=8, n_classes=4, random_state=seed)
        try:
            func = getattr(submission_module, 'stratified_group_k_fold', None)
            if func is None:
                results.append(('error', 'missing_func'))
                continue
            out = func(df.copy(), group_col='group', label_col='label', n_splits=n_splits, random_state=seed)
        except Exception as e:
            tb = traceback.format_exc()
            results.append(('error', 'exception', str(e), tb))
            continue

        # determinism check: same seed should yield identical structure if recomputed
        try:
            out_repeat = func(df.copy(), group_col='group', label_col='label', n_splits=n_splits, random_state=seed)
        except Exception:
            out_repeat = None

        def structure_hash(o):
            if isinstance(o, list):
                return tuple(sorted([(tuple(sorted(train_idx.tolist())), tuple(sorted(val_idx.tolist()))) for train_idx, val_idx in o]))
            elif isinstance(o, pd.DataFrame):
                return tuple(o.sort_index()['cv_fold'].tolist())
            else:
                return None
        if structure_hash(out) != structure_hash(out_repeat):
            results.append(('fail', 'not_deterministic'))
            continue

        # normalize output to assignments and perform checks
        if isinstance(out, list):
            # Basic shape validation
            if len(out) != n_splits:
                results.append(('fail', 'wrong_number_folds'))
                continue
            # ensure non-empty val sets
            if any(len(val_idx) == 0 for _, val_idx in out):
                results.append(('fail', 'empty_validation_fold'))
                continue
            # group leakage
            ok, reason = no_group_leakage(df, out, 'group')
            if not ok:
                results.append(('fail', reason))
                continue
            # overlapping indices between folds' validation sets
            val_union = []
            for _, val_idx in out:
                val_union.extend(val_idx.tolist())
            if len(val_union) != len(set(val_union)):
                results.append(('fail', 'duplicate_validation_indices'))
                continue
            # stratification tolerance & label presence
            ok2, reason2 = stratification_ok(df, out, 'label')
            if not ok2:
                results.append(('fail', reason2))
                continue
            results.append(('pass', 'ok'))
        elif isinstance(out, pd.DataFrame):
            if 'cv_fold' not in out.columns:
                results.append(('fail', 'no_cv_fold'))
                continue
            # verify fold count
            unique_folds = sorted(out['cv_fold'].unique())
            if len(unique_folds) != n_splits or unique_folds != list(range(n_splits)):
                results.append(('fail', 'incorrect_fold_labels'))
                continue
            # ensure non-empty validation sets
            if any((out['cv_fold'] == f).sum() == 0 for f in unique_folds):
                results.append(('fail', 'empty_validation_fold'))
                continue
            # group leakage
            ok, reason = no_group_leakage(df, out, 'group')
            if not ok:
                results.append(('fail', reason))
                continue
            # index duplication between folds (should be partition of dataset)
            if sorted(out.index.tolist()) != sorted(df.index.tolist()):
                results.append(('fail', 'index_mismatch'))
                continue
            # stratification tolerance & label presence
            ok2, reason2 = stratification_ok(df, out, 'label')
            if not ok2:
                results.append(('fail', reason2))
                continue
            results.append(('pass', 'ok'))
        else:
            results.append(('fail', 'bad_return'))
    # compute pass rate
    pass_count = sum(1 for r in results if r[0] == 'pass')
    pass_rate = pass_count / len(results)
    return pass_rate, results

# --- utility: import student file from path ---
def load_submission_from_path(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = module
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    # quick test: load local submission.py
    mod = load_submission_from_path("submission.py")
    rate, details = grade_submission(mod, num_trials=10, n_splits=4)
    print("pass_rate", rate)
    for d in details:
        print(d)
