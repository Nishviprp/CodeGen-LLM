# evaluation/evaluator.py
"""
WHAT THIS FILE DOES:
====================
Runs both models on all 164 HumanEval tasks and computes four metrics.

THE EVALUATION LOOP (per task):
  1. Feed the task's prompt to both models
  2. Generate 5 completions each (for pass@k)
  3. For each completion: prepend prompt, append test code, run in subprocess
  4. Record pass/fail, timing, complexity, ROUGE similarity
  5. Compute pass@1 and pass@5 using the unbiased estimator

THE METRICS EXPLAINED:

  pass@k  ─────────────────────────────────────────────────────────────────
  The main HumanEval metric. Instead of "did it work once?", it asks:
  "if you tried k times, what's the probability at least one works?"

  We generate n=5 samples and use the Chen et al. 2021 formula:
    pass@k = 1 - C(n-c, k) / C(n, k)
  where c = number of passing samples, n = total samples.
  This is UNBIASED — it doesn't overcount lucky single successes.

  Code Complexity  ────────────────────────────────────────────────────────
  Two simple proxies:
    - Line count: longer code isn't always better (Copilot scores here)
    - Control structures: count of if/for/while/try etc. per function
  A model that writes 50-line functions for simple tasks is generating noise.

  Generation Latency  ─────────────────────────────────────────────────────
  Wall-clock seconds per completion. Fine-tuned models run locally so
  latency depends on your hardware. Baseline downloads ~3GB on first run
  but then runs locally too.

  ROUGE-L Similarity  ─────────────────────────────────────────────────────
  Measures token overlap between generated code and the reference solution.
  Important caveat: high ROUGE ≠ correct code. Two completely different
  implementations of the same function can both be correct but have low
  ROUGE. We include it for continuity with prior work but do NOT use it
  as a primary metric.

IMPORTANT IMPLEMENTATION NOTE:
  When evaluating, we do:
    full_program = prompt + completion + "\n\n" + test_code
  This is necessary because models only generate the completion (function body).
  The test code calls check(entry_point), which needs the full function definition.
  Forgetting to prepend the prompt is a common bug that causes IndentationError
  on every single task — we explicitly guard against this.

OUTPUT:
  results/detailed_results.csv  — per-task rows with all metrics
  results/summary_results.csv   — mean metrics across all 164 tasks
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.codegen_model import FineTunedCodeGen
from models.baseline_model import QwenCoderBaseline


# ─── pass@k estimator (Chen et al. 2021) ─────────────────────────────────────

def pass_at_k(n, c, k):
    """
    Unbiased estimator for pass@k.
    n = total completions generated per task
    c = how many of those passed all tests
    k = k in pass@k

    Special case: if all n completions passed, return 1.0 immediately
    (avoids a division-by-zero in the combinatorial term).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ─── Test execution harness ───────────────────────────────────────────────────

def run_test(prompt, completion, test_code, entry_point, timeout=5):
    """
    Assembles the full program and runs it in an isolated subprocess.

    Full program structure:
        [prompt]           — function signature + docstring
        [completion]       — generated function body
        [test_code]        — def check(candidate): assert ...
        check(entry_point) — actually calls check() with our function

    Returns dict with:
        success: bool
        stderr:  error message if failed (truncated to 300 chars)
        error_type: TIMEOUT | SYNTAX | RUNTIME | ASSERTION | OK
    """
    full_program = (
        prompt
        + completion
        + "\n\n"
        + test_code
        + f"\n\ncheck({entry_point})\n"
    )

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(full_program)
        tmp = f.name

    try:
        result = subprocess.run(
            ["python", tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        stderr = result.stderr[-300:] if result.stderr else ""

        # Categorize the failure type (useful for analysis)
        if success:
            error_type = "OK"
        elif "SyntaxError" in stderr or "IndentationError" in stderr:
            error_type = "SYNTAX"
        elif "AssertionError" in stderr:
            error_type = "ASSERTION"
        else:
            error_type = "RUNTIME"

        return {"success": success, "stderr": stderr, "error_type": error_type}

    except subprocess.TimeoutExpired:
        return {"success": False, "stderr": "Execution timed out", "error_type": "TIMEOUT"}
    except Exception as e:
        return {"success": False, "stderr": str(e), "error_type": "RUNTIME"}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ─── Metric helpers ───────────────────────────────────────────────────────────

def code_complexity(code):
    """Count lines and control flow keywords as a complexity proxy."""
    if not code or not code.strip():
        return {"lines": 0, "control_structures": 0}
    lines = len(code.strip().split("\n"))
    keywords = ["if ", "elif ", "else:", "for ", "while ", "try:", "except", "with ", "def ", "class "]
    controls = sum(code.count(kw) for kw in keywords)
    return {"lines": lines, "control_structures": controls}


_rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def rouge_similarity(generated, reference):
    """ROUGE-L F1 between generated code and reference solution."""
    if not generated or not generated.strip():
        return {"rouge1": 0.0, "rougeL": 0.0}
    scores = _rouge.score(reference, generated)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(
    data_path    = "data/human_eval.json",
    output_dir   = "results",
    num_tasks    = 164,       # full HumanEval set
    n_per_task   = 5,         # completions per model per task → enables pass@1 and pass@5
    max_tokens   = 192,
    temperature  = 0.6,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load eval data
    with open(data_path) as f:
        tasks = json.load(f)
    tasks = tasks[:num_tasks]
    print(f"Evaluating on {len(tasks)} HumanEval tasks, {n_per_task} samples/task/model\n")

    # Load models
    model_a = FineTunedCodeGen()
    model_b = QwenCoderBaseline()

    rows = []

    for idx, task in enumerate(tasks):
        task_id     = task["task_id"]
        prompt      = task["prompt"]
        reference   = task["canonical_solution"]
        test_code   = task["test"]
        entry_point = task["entry_point"]

        print(f"[{idx+1:3d}/{len(tasks)}] {task_id}")

        # Generate n completions per model and test each
        a_results, b_results = [], []

        for _ in range(n_per_task):
            # Model A
            t0 = time.time()
            try:
                comp_a = model_a.generate_code(prompt, max_new_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                comp_a = ""
            time_a = time.time() - t0
            test_a = run_test(prompt, comp_a, test_code, entry_point)
            a_results.append({**test_a, "code": comp_a, "time": time_a})

            # Model B
            t0 = time.time()
            try:
                comp_b = model_b.generate_code(prompt, max_new_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                comp_b = ""
            time_b = time.time() - t0
            test_b = run_test(prompt, comp_b, test_code, entry_point)
            b_results.append({**test_b, "code": comp_b, "time": time_b})

        # pass@k using the first completion for complexity/ROUGE (rest are for pass@k stats)
        n = n_per_task
        c_a = sum(r["success"] for r in a_results)
        c_b = sum(r["success"] for r in b_results)

        a_first_code = a_results[0]["code"]
        b_first_code = b_results[0]["code"]

        rows.append({
            "task_id":           task_id,
            # pass@k
            "codegen_pass@1":    round(pass_at_k(n, c_a, 1), 4),
            "qwen_pass@1":       round(pass_at_k(n, c_b, 1), 4),
            "codegen_pass@5":    round(pass_at_k(n, c_a, min(5, n)), 4),
            "qwen_pass@5":       round(pass_at_k(n, c_b, min(5, n)), 4),
            "codegen_correct":   c_a,
            "qwen_correct":      c_b,
            # failure breakdown
            "codegen_error_types": str([r["error_type"] for r in a_results if not r["success"]]),
            "qwen_error_types":    str([r["error_type"] for r in b_results if not r["success"]]),
            # latency
            "codegen_avg_time":  round(np.mean([r["time"] for r in a_results]), 3),
            "qwen_avg_time":     round(np.mean([r["time"] for r in b_results]), 3),
            # complexity (first sample)
            "codegen_lines":     code_complexity(a_first_code)["lines"],
            "qwen_lines":        code_complexity(b_first_code)["lines"],
            "codegen_controls":  code_complexity(a_first_code)["control_structures"],
            "qwen_controls":     code_complexity(b_first_code)["control_structures"],
            # ROUGE (first sample vs reference)
            "codegen_rouge1":    rouge_similarity(a_first_code, reference)["rouge1"],
            "qwen_rouge1":       rouge_similarity(b_first_code, reference)["rouge1"],
            "codegen_rougeL":    rouge_similarity(a_first_code, reference)["rougeL"],
            "qwen_rougeL":       rouge_similarity(b_first_code, reference)["rougeL"],
            # sample code (for qualitative analysis)
            "codegen_sample":    a_first_code[:500],
            "qwen_sample":       b_first_code[:500],
        })

    # Save per-task results
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)

    # Save summary
    summary = {
        "codegen_pass@1_mean":  df["codegen_pass@1"].mean(),
        "qwen_pass@1_mean":     df["qwen_pass@1"].mean(),
        "codegen_pass@5_mean":  df["codegen_pass@5"].mean(),
        "qwen_pass@5_mean":     df["qwen_pass@5"].mean(),
        "codegen_avg_lines":    df["codegen_lines"].mean(),
        "qwen_avg_lines":       df["qwen_lines"].mean(),
        "codegen_avg_controls": df["codegen_controls"].mean(),
        "qwen_avg_controls":    df["qwen_controls"].mean(),
        "codegen_avg_time":     df["codegen_avg_time"].mean(),
        "qwen_avg_time":        df["qwen_avg_time"].mean(),
        "codegen_avg_rouge1":   df["codegen_rouge1"].mean(),
        "qwen_avg_rouge1":      df["qwen_rouge1"].mean(),
        "codegen_avg_rougeL":   df["codegen_rougeL"].mean(),
        "qwen_avg_rougeL":      df["qwen_rougeL"].mean(),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)

    print("\n========== RESULTS SUMMARY ==========")
    print(f"Tasks evaluated : {len(df)}")
    print(f"Samples/task    : {n_per_task}")
    print(f"\n{'Metric':<28} {'CodeGen-350M':>14} {'Qwen2.5-1.5B':>14}")
    print("-" * 58)
    for k in ["pass@1_mean", "pass@5_mean"]:
        print(f"  {k:<26} {summary[f'codegen_{k}']:>14.4f} {summary[f'qwen_{k}']:>14.4f}")
    print(f"  {'avg_time (s)':<26} {summary['codegen_avg_time']:>14.3f} {summary['qwen_avg_time']:>14.3f}")
    print(f"  {'avg_lines':<26} {summary['codegen_avg_lines']:>14.1f} {summary['qwen_avg_lines']:>14.1f}")
    print(f"  {'avg_rougeL':<26} {summary['codegen_avg_rougeL']:>14.4f} {summary['qwen_avg_rougeL']:>14.4f}")
    print("=====================================")
    print(f"\nDetailed results → {output_dir}/detailed_results.csv")
    print(f"Summary          → {output_dir}/summary_results.csv")
    print("Next step: python evaluation/visualize.py")

    return df, summary_df


if __name__ == "__main__":
    evaluate()
