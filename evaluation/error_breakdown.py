"""
error_breakdown.py
Qualitative failure-mode analysis for the CodeGen-350M vs Qwen2.5-Coder-1.5B
HumanEval comparison.

Reads results/detailed_results.csv (produced by evaluator.py), classifies every
FAILED completion into the harness's four execution buckets, and quantifies the
output-format confound that inflates the instruction-tuned model's syntax-failure
count.

Buckets (assigned in evaluator.run_test):
  SYNTAX    -> SyntaxError / IndentationError: did not yield executable Python
  ASSERTION -> AssertionError: ran, but failed the unit test  (valid code, wrong logic)
  RUNTIME   -> other uncaught exception during execution
  TIMEOUT   -> exceeded the per-task execution timeout

Outputs:
  results/error_breakdown.csv          failure-type table (counts + % of failures)
  results/visualizations/error_breakdown.png   grouped bar figure
"""
import ast
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "detailed_results.csv"
OUT_CSV = ROOT / "results" / "error_breakdown.csv"
OUT_PNG = ROOT / "results" / "visualizations" / "error_breakdown.png"

CATS = ["SYNTAX", "ASSERTION", "RUNTIME", "TIMEOUT"]
PRETTY = {
    "SYNTAX": "Syntax\n(no valid code)",
    "ASSERTION": "Assertion\n(wrong logic)",
    "RUNTIME": "Runtime\n(crash)",
    "TIMEOUT": "Timeout",
}
N_PER_TASK = 5


def tally(series):
    c = Counter()
    for v in series.dropna():
        for lbl in ast.literal_eval(v):
            c[lbl] += 1
    return c


def parses_standalone(code):
    try:
        ast.parse(str(code).replace("\\n", "\n"))
        return True
    except Exception:
        return False


def main():
    df = pd.read_csv(CSV)
    n_total = len(df) * N_PER_TASK

    cg, qw = tally(df["codegen_error_types"]), tally(df["qwen_error_types"])
    cg_f, qw_f = sum(cg.values()), sum(qw.values())

    table = pd.DataFrame(
        {
            "failure_type": CATS,
            "codegen_count": [cg.get(k, 0) for k in CATS],
            "codegen_pct_of_failures": [round(cg.get(k, 0) / cg_f * 100, 1) for k in CATS],
            "qwen_count": [qw.get(k, 0) for k in CATS],
            "qwen_pct_of_failures": [round(qw.get(k, 0) / qw_f * 100, 1) for k in CATS],
        }
    )
    table.to_csv(OUT_CSV, index=False)

    # Confound diagnostic: valid standalone code that still scored as a failure.
    qok = df["qwen_sample"].dropna().astype(str).apply(parses_standalone)
    cok = df["codegen_sample"].dropna().astype(str).apply(parses_standalone)
    valid_but_failed = int(
        ((qok) & (df.loc[qok.index, "qwen_pass@1"] == 0.0)).sum()
    )

    print(f"Completions per model: {n_total}")
    print(f"CodeGen failures: {cg_f} (pass {n_total-cg_f}) | Qwen failures: {qw_f} (pass {n_total-qw_f})")
    print(table.to_string(index=False))
    print(f"\nQwen completion-1 parses standalone: {qok.mean()*100:.1f}%")
    print(f"CodeGen completion-1 parses standalone: {cok.mean()*100:.1f}%")
    print(f"Qwen tasks: valid standalone code yet pass@1==0: {valid_but_failed}/{len(df)}")

    # ---- figure ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=160)
    x = range(len(CATS))
    w = 0.38
    cg_vals = [cg.get(k, 0) for k in CATS]
    qw_vals = [qw.get(k, 0) for k in CATS]
    b1 = ax.bar([i - w / 2 for i in x], cg_vals, w, label="CodeGen-350M (fine-tuned)", color="#c44e52")
    b2 = ax.bar([i + w / 2 for i in x], qw_vals, w, label="Qwen2.5-Coder-1.5B-Instruct", color="#4c72b0")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 6, str(int(h)),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([PRETTY[k] for k in CATS])
    ax.set_ylabel("Failed completions (of 820 per model)")
    ax.set_title("HumanEval failure-type distribution by model")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(max(cg_vals), max(qw_vals)) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"\nWrote {OUT_CSV.name} and {OUT_PNG.name}")


if __name__ == "__main__":
    main()
