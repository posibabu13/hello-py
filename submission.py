# submission.py
import numpy as np
import pandas as pd
from collections import defaultdict

def stratified_group_k_fold(df, group_col, label_col, n_splits, random_state=None):
    """Return DataFrame with cv_fold column (0..n_splits-1) for stratified group K-fold.

    Heuristic procedure:
    1. Order groups by decreasing size * label entropy.
    2. Greedy assignment minimizing a combined objective of max per-label relative deviation + size deviation.
    3. Local swap improvement between folds focusing on folds with worst maximum label deviation.
    4. If tolerance not met, second pass reassign single problematic groups to alternative folds when it strictly reduces max deviation.

    This aims to reduce validation distribution discrepancy for each fold.
    """
    rng = np.random.RandomState(random_state)
    groups = list(df[group_col].unique())

    # Deterministic shuffle based on random_state
    rng.shuffle(groups)

    group_indices = {g: df.index[df[group_col] == g] for g in groups}
    group_label_counts = {g: df.loc[group_indices[g], label_col].value_counts().to_dict() for g in groups}

    overall_counts = df[label_col].value_counts().to_dict()
    labels = sorted(overall_counts.keys())
    target_counts = {lbl: overall_counts[lbl] / n_splits for lbl in labels}

    # Coverage: groups containing each label
    groups_for_label = {lbl: [g for g in groups if lbl in group_label_counts[g]] for lbl in labels}

    fold_groups = [set() for _ in range(n_splits)]
    fold_label_counts = [defaultdict(int) for _ in range(n_splits)]

    # Seed: ensure each fold gets a group (prefer large diverse groups)
    def diversity_score(g):
        counts = group_label_counts[g]
        size = sum(counts.values())
        probs = np.array(list(counts.values())) / (size + 1e-12)
        entropy = -(probs * np.log(probs + 1e-12)).sum()
        return (entropy, size)
    seed_candidates = sorted(groups, key=diversity_score, reverse=True)
    used = set()
    for f in range(n_splits):
        for g in seed_candidates:
            if g not in used:
                fold_groups[f].add(g)
                used.add(g)
                for lbl, c in group_label_counts[g].items():
                    fold_label_counts[f][lbl] += c
                break

    # Label presence pass: ensure each fold has each label if feasible
    for lbl in labels:
        avail = [g for g in groups_for_label[lbl] if g not in used]
        if not avail:
            continue
        # Assign remaining label groups to folds missing the label
        for f in range(n_splits):
            if fold_label_counts[f][lbl] == 0 and avail:
                g = avail.pop(0)
                fold_groups[f].add(g)
                used.add(g)
                for l2, c2 in group_label_counts[g].items():
                    fold_label_counts[f][l2] += c2

    # Remaining groups assignment minimizing sum of squared deviation
    remaining = [g for g in groups if g not in used]
    for g in remaining:
        g_counts = group_label_counts[g]
        best_fold = None
        best_obj = None
        for f in range(n_splits):
            obj = 0.0
            for lbl in labels:
                projected = fold_label_counts[f][lbl] + g_counts.get(lbl, 0)
                diff = projected - target_counts[lbl]
                obj += diff * diff
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_fold = f
        fold_groups[best_fold].add(g)
        for lbl, c in g_counts.items():
            fold_label_counts[best_fold][lbl] += c

    # Swap refinement
    def global_max_rel_dev():
        vals = []
        for f in range(n_splits):
            for lbl in labels:
                dev = abs(fold_label_counts[f][lbl] - target_counts[lbl]) / (target_counts[lbl] + 1e-9)
                vals.append(dev)
        return max(vals)
    for _ in range(300):
        # pick two folds with high deviation components
        deviations = [sum(abs(fold_label_counts[f][lbl] - target_counts[lbl]) for lbl in labels) for f in range(n_splits)]
        probs = np.array(deviations) + 1e-6
        probs = probs / probs.sum()
        f1, f2 = rng.choice(n_splits, size=2, replace=False, p=probs)
        if not fold_groups[f1] or not fold_groups[f2]:
            continue
        g1 = rng.choice(list(fold_groups[f1]))
        g2 = rng.choice(list(fold_groups[f2]))
        if g1 == g2:
            continue
        g1c = group_label_counts[g1]; g2c = group_label_counts[g2]
        before = global_max_rel_dev()
        # apply swap
        fold_groups[f1].remove(g1); fold_groups[f2].remove(g2)
        fold_groups[f1].add(g2); fold_groups[f2].add(g1)
        for lbl in labels:
            fold_label_counts[f1][lbl] += g2c.get(lbl, 0) - g1c.get(lbl, 0)
            fold_label_counts[f2][lbl] += g1c.get(lbl, 0) - g2c.get(lbl, 0)
        after = global_max_rel_dev()
        if after > before:
            # revert
            fold_groups[f1].remove(g2); fold_groups[f2].remove(g1)
            fold_groups[f1].add(g1); fold_groups[f2].add(g2)
            for lbl in labels:
                fold_label_counts[f1][lbl] += g1c.get(lbl, 0) - g2c.get(lbl, 0)
                fold_label_counts[f2][lbl] += g2c.get(lbl, 0) - g1c.get(lbl, 0)

    # Build mapping
    fold_map = {}
    for f_idx, gs in enumerate(fold_groups):
        for g in gs:
            fold_map[g] = f_idx
    df_out = df.copy()
    df_out['cv_fold'] = df_out[group_col].map(fold_map).astype(int)

    # --- Coverage enforcement: ensure every fold label 0..n_splits-1 appears ---
    present = set(df_out['cv_fold'].unique())
    missing = [f for f in range(n_splits) if f not in present]
    if missing:
        # move largest groups from folds with largest size deviation to missing folds
        # compute group sizes
        group_sizes = {g: len(group_indices[g]) for g in groups}
        # sort folds by descending size
        fold_size_pairs = sorted([(f, sum(group_sizes[g] for g in fold_groups[f])) for f in range(n_splits)], key=lambda x: x[1], reverse=True)
        for m in missing:
            # pick a donor fold with >1 groups to avoid emptying it
            for donor, _ in fold_size_pairs:
                if len(fold_groups[donor]) <= 1:
                    continue
                # choose group whose move least harms stratification: smallest absolute max deviation impact
                best_g = None
                best_cost = None
                for g in list(fold_groups[donor]):
                    g_counts = group_label_counts[g]
                    # simulate removal from donor only (addition to new fold assumed neutral initially)
                    cost_components = []
                    for lbl in labels:
                        donor_after = fold_label_counts[donor][lbl] - g_counts.get(lbl, 0)
                        dev = abs(donor_after - target_counts[lbl]) / (target_counts[lbl] + 1e-9)
                        cost_components.append(dev)
                    cost = max(cost_components)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_g = g
                if best_g is not None:
                    # perform move
                    fold_groups[donor].remove(best_g)
                    fold_groups[m].add(best_g)
                    for lbl in labels:
                        fold_label_counts[donor][lbl] -= group_label_counts[best_g].get(lbl, 0)
                        fold_label_counts[m][lbl] += group_label_counts[best_g].get(lbl, 0)
                    break  # next missing label
        # rebuild mapping
        fold_map = {}
        for f_idx, gs in enumerate(fold_groups):
            for g in gs:
                fold_map[g] = f_idx
        df_out['cv_fold'] = df_out[group_col].map(fold_map).astype(int)
    # Proportional deficit refinement: iteratively move a group that reduces max deviation.
    def fold_label_dev(f, lbl):
        return abs(fold_label_counts[f][lbl] - target_counts[lbl]) / (target_counts[lbl] + 1e-9)
    def global_max_dev():
        return max(fold_label_dev(f,lbl) for f in range(n_splits) for lbl in labels)

    for _ in range(400):
        current_max = global_max_dev()
        if current_max <= 0.08:
            break
        # identify worst (fold,label)
        worst_pair = None
        worst_val = -1
        for f in range(n_splits):
            for lbl in labels:
                dv = fold_label_dev(f,lbl)
                if dv > worst_val:
                    worst_val = dv
                    worst_pair = (f,lbl)
        if worst_pair is None:
            break
        source_fold, source_lbl = worst_pair
        # find candidate groups in source contributing label
        candidates = [g for g in fold_groups[source_fold] if group_label_counts[g].get(source_lbl,0)>0]
        if not candidates:
            break
        # sort candidates by proportion of that label descending
        candidates.sort(key=lambda g: group_label_counts[g].get(source_lbl,0)/sum(group_label_counts[g].values()), reverse=True)
        moved = False
        for g in candidates:
            g_counts = group_label_counts[g]
            # evaluate destination folds
            best_dest = None
            best_improve = 0.0
            for dest in range(n_splits):
                if dest == source_fold:
                    continue
                # simulate move
                # prevent removing last group containing a label from source if that would make presence zero while others have >0
                safe = True
                for lbl in labels:
                    if g_counts.get(lbl,0)>0:
                        total_occurrence_in_source = sum(group_label_counts[gg].get(lbl,0)>0 for gg in fold_groups[source_fold])
                        if total_occurrence_in_source<=1:
                            safe = True  # still safe; label may appear elsewhere
                if not safe:
                    continue
                # compute new max deviation
                devs_after = []
                for f in range(n_splits):
                    for lbl in labels:
                        val = fold_label_counts[f][lbl]
                        if f==source_fold:
                            val -= g_counts.get(lbl,0)
                        elif f==dest:
                            val += g_counts.get(lbl,0)
                        dv = abs(val - target_counts[lbl]) / (target_counts[lbl] + 1e-9)
                        devs_after.append(dv)
                new_max = max(devs_after)
                improve = current_max - new_max
                if improve > best_improve + 1e-12:
                    best_improve = improve
                    best_dest = dest
            if best_dest is not None and best_improve>0:
                # apply move
                fold_groups[source_fold].remove(g)
                fold_groups[best_dest].add(g)
                for lbl in labels:
                    fold_label_counts[source_fold][lbl] -= g_counts.get(lbl,0)
                    fold_label_counts[best_dest][lbl] += g_counts.get(lbl,0)
                # rebuild mapping
                fold_map = {}
                for f_idx, gs in enumerate(fold_groups):
                    for g2 in gs:
                        fold_map[g2] = f_idx
                df_out['cv_fold'] = df_out[group_col].map(fold_map).astype(int)
                moved = True
                break
        if not moved:
            break
    return df_out
