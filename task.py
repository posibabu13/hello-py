import asyncio
import json
from io import StringIO
from contextlib import redirect_stdout
import types
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from grader import grade_submission, make_synthetic_df

# Persistent namespace for incremental code authoring across tool calls
PERSISTENT_NS: dict[str, object] = {}


# ------------------------
# Tools
# ------------------------

def python_expression_tool(expression: str):
    """Executes Python code in a persistent namespace and returns stdout.

    The model can build up the implementation of stratified_group_k_fold over multiple calls.
    """
    try:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, PERSISTENT_NS, PERSISTENT_NS)
        return {"result": stdout.getvalue(), "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: int):
    """Marks answer as submitted."""
    return {"answer": answer, "submitted": True}


# ------------------------
# Agent loop
# ------------------------

async def run_agent(prompt, tools, handlers, max_steps=8):
    """Runs the agent and returns True when the model signals completion via submit_answer."""
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            tools=tools,
            messages=messages,
        )

        submitted = False
        tool_results = []
        has_tool_call = False

        for content in response.content:
            if content.type == "tool_use":
                has_tool_call = True
                tool_name = content.name
                tool_input = content.input
                handler = handlers[tool_name]
                if tool_name == "python_expression":
                    result = handler(tool_input["expression"])
                elif tool_name == "submit_answer":
                    result = handler(tool_input["answer"])
                    submitted = True
                else:
                    result = {"result": None, "error": f"Unknown tool {tool_name}"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content.id,
                    "content": json.dumps(result),
                })
        if has_tool_call:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            if submitted:
                return True
        else:
            break
    return False


# ------------------------
# Grading logic
# ------------------------

async def grade():
    tools = [
        {
            "name": "python_expression",
            "description": "Executes Python code; build your function incrementally; use print for inspection.",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Signal you are done implementing stratified_group_k_fold",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    ]

    handlers = {"python_expression": python_expression_tool, "submit_answer": submit_answer_tool}

    prompt = (
        "Implement stratified_group_k_fold(df, group_col, label_col, n_splits, random_state=None).\n\n"
        "Return EITHER: (a) list of length n_splits of (train_idx, val_idx) numpy index arrays OR (b) DataFrame == df with added cv_fold column in 0..n_splits-1 assigning each row to EXACTLY one validation fold.\n\n"
        "Requirements (all enforced):\n"
        "R1 Determinism: same inputs + random_state yield identical structure.\n"
        "R2 Fold count/labels: exactly n_splits folds; DataFrame form must have labels 0..n_splits-1.\n"
        "R3 Non-empty validation per fold.\n"
        "R4 Group isolation: no group appears in both train and validation for a fold.\n"
        "R5 Partitioning: validation sets collectively partition df (no duplicates; DataFrame preserves all indices).\n"
        "R6 Stratification dual tolerance: for EACH fold: max per-class relative frequency deviation ≤ 0.13 AND mean deviation ≤ 0.06.\n"
        "R7 Label presence: every global label appears ≥1 time in each validation fold (if possible given grouping).\n"
        "R8 Return type validity: structure matches one accepted form precisely.\n"
        "Advisory: Large fold size imbalance is tolerated (soft check).\n"
        "Notes: Preserve df.index; may use numpy/pandas/sklearn; avoid I/O. Strategy choices (greedy, optimization, refinement) are all acceptable. Submit when ready using submit_answer."
    )

    runs = 10
    successes = 0
    pass_rates = []
    for i in range(runs):
        # clear namespace each trial so model must rebuild
        PERSISTENT_NS.clear()
        print(f"\n=== Trial {i+1}/{runs} ===")
        finished = await run_agent(prompt, tools, handlers)
        func = PERSISTENT_NS.get("stratified_group_k_fold")
        if not callable(func):
            print("Missing function; automatic failure.")
            pass_rate = 0.0
            pass_rates.append(pass_rate)
            continue
        # Wrap into a pseudo module for grader
        mod = types.SimpleNamespace(stratified_group_k_fold=func)
        pass_rate, details = grade_submission(mod, num_trials=10, n_splits=4)
        pass_rates.append(pass_rate)
        # Consider trial success if pass_rate in target window 0.10–0.40
        trial_success = 0.10 <= pass_rate <= 0.40
        successes += int(trial_success)
        print(f"Trial pass_rate={pass_rate:.2f} success={trial_success}")
        # Show a compact failure reason histogram
        from collections import Counter
        reason_counts = Counter(r[1] for r in details if r[0] != 'pass')
        if reason_counts:
            top = ', '.join(f"{k}:{v}" for k, v in list(reason_counts.items())[:6])
            print(f"Failure reasons: {top}")
    overall_rate = successes / runs
    print("\nOverall success fraction (trials in window 0.10–0.40):", overall_rate)
    print("Per-trial pass rates:", [f"{r:.2f}" for r in pass_rates])
    return overall_rate


if __name__ == "__main__":
    asyncio.run(grade())
