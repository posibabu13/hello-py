hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Stratified Group K-Fold RL Task

The repository now includes an RL-style task (`task.py`) where a model must implement `stratified_group_k_fold` achieving a pass rate within a target window.

### Objective
Write `stratified_group_k_fold(df, group_col, label_col, n_splits, random_state=None)` that returns either:
1. A list of length `n_splits` containing `(train_idx, val_idx)` numpy index arrays, OR
2. A DataFrame equal to the input with an added `cv_fold` column (integer labels `0..n_splits-1`).

### Enforced Requirements
R1 Determinism (same seed → identical structure)
R2 Correct fold count / labels
R3 Non-empty validation fold
R4 Group isolation (no group leaked train↔validation within a fold)
R5 Partitioning (validation sets collectively partition dataset; no duplicates; DataFrame form preserves all indices)
R6 Dual stratification tolerance: per fold max class frequency deviation ≤ 0.13 AND mean deviation ≤ 0.06
R7 Label presence (every global label appears ≥1 time per validation fold if feasible)
R8 Return type validity (exactly one accepted form)

Fold size imbalance is currently a soft advisory (non-failing) check to avoid over-constraining solutions.

### Pass Rate Target
The grader (`grader.py`) runs multiple seeds and computes the fraction of seeds passing. For RL evaluation in `task.py`, a trial is considered successful if its internal pass rate lies within `[0.10, 0.40]`, ensuring the task is neither trivial nor impossible.

### Multiple Strategies Encouraged
- Greedy assignment of groups minimizing deviation
- Iterative refinement / swapping groups
- Optimization using target per-fold label proportions
- Hybrid heuristics

### Running the RL Task
```
uv run task.py
```
This invokes an agent loop requiring tool calls to build the function (using a persistent Python namespace) and then submit completion.

## Anthropic API Key Persistence
Set once per shell session:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```
For permanence add to `~/.zshrc`:
```bash
echo 'export ANTHROPIC_API_KEY=your_api_key_here' >> ~/.zshrc
source ~/.zshrc
```
Verify:
```bash
echo $ANTHROPIC_API_KEY
```

## Troubleshooting
- If `Import "anthropic" could not be resolved`: ensure dependency installed (see `pyproject.toml`).
- Determinism failures: check usage of `random_state` and avoid unintended global RNG calls.
- Stratification failures (`strat_max_fail_fold_X` / `strat_mean_fail_fold_X`): inspect per-fold label distributions versus global proportions.

## Calibration Results

Recent grader run (10 seeds, n_splits=4):

```
pass_rate 0.2
('fail', 'strat_max_fail_fold_3')
('fail', 'strat_max_fail_fold_2')
('pass', 'ok')
('fail', 'strat_mean_fail_fold_0')
('fail', 'strat_max_fail_fold_0')
('fail', 'strat_max_fail_fold_1')
('fail', 'strat_max_fail_fold_2')
('pass', 'ok')
('fail', 'strat_mean_fail_fold_0')
('fail', 'strat_max_fail_fold_0')
```


Current tolerances:
- Max per-class relative frequency deviation ≤ 0.132
- Mean per-class relative frequency deviation ≤ 0.065

Second sample (different 10-seed run):

```
pass_rate 0.2
('fail', 'strat_mean_fail_fold_2')
('fail', 'strat_mean_fail_fold_2')
('pass', 'ok')
('fail', 'strat_max_fail_fold_0')
('fail', 'strat_max_fail_fold_3')
('fail', 'strat_mean_fail_fold_1')
('fail', 'strat_max_fail_fold_0')
('pass', 'ok')
('fail', 'strat_mean_fail_fold_0')
('fail', 'strat_max_fail_fold_0')
```

Observation: Distribution of failure reasons shifts across runs (more mean deviations in this sample), reinforcing that the task exposes multiple distinct challenges rather than a single brittle check.

\n+Third sample (temporary relaxed tolerances max_tol=0.135, mean_tol=0.07 to illustrate effect on difficulty):

```
pass_rate 0.5
('pass', 'ok')
('fail', 'strat_max_fail_fold_2')
('pass', 'ok')
('fail', 'strat_max_fail_fold_0')
('fail', 'strat_mean_fail_fold_0')
('fail', 'strat_mean_fail_fold_1')
('pass', 'ok')
('pass', 'ok')
('fail', 'strat_max_fail_fold_0')
('pass', 'ok')
```
This demonstrates calibration sensitivity: slight relaxation raises pass_rate above the target window (now 50%), confirming chosen production tolerances (0.132 / 0.065) keep the task challenging without being impossible.


## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.
