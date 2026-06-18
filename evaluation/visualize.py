# evaluation/visualize.py
"""
WHAT THIS FILE DOES:
====================
Reads results/detailed_results.csv and produces 5 publication-ready charts.

WHY A SEPARATE FILE?
  You can re-run this without re-running the expensive 90-minute evaluation.
  Tweak colors, labels, or add new charts here without touching the eval logic.

CHARTS PRODUCED:
  1. pass_at_k.png          — bar chart: pass@1 and pass@5 for both models
  2. latency.png            — bar chart: average generation time per model
  3. complexity_lines.png   — bar chart: average line count per model
  4. complexity_controls.png— bar chart: average control structures per model
  5. rouge_scores.png       — grouped bar chart: ROUGE-1 and ROUGE-L
  6. per_task_pass1.png     — line plot: pass@1 per task (shows which tasks are hard)

All saved to results/visualizations/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works on Colab and servers)


RESULTS_PATH   = "results/detailed_results.csv"
SUMMARY_PATH   = "results/summary_results.csv"
OUTPUT_DIR     = "results/visualizations"

# Color scheme: consistent across all charts
COLOR_A = "#2563EB"   # blue  → fine-tuned CodeGen
COLOR_B = "#F59E0B"   # amber → Qwen2.5-Coder baseline

LABEL_A = "CodeGen-350M\n(fine-tuned)"
LABEL_B = "Qwen2.5-Coder\n1.5B-Instruct"


def load_data():
    df      = pd.read_csv(RESULTS_PATH)
    summary = pd.read_csv(SUMMARY_PATH).iloc[0]
    return df, summary


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def chart_pass_at_k(summary):
    """Bar chart comparing pass@1 and pass@5 for both models."""
    fig, ax = plt.subplots(figsize=(7, 5))

    metrics = ["pass@1", "pass@5"]
    x = range(len(metrics))
    width = 0.3

    vals_a = [summary["codegen_pass@1_mean"], summary["codegen_pass@5_mean"]]
    vals_b = [summary["qwen_pass@1_mean"],    summary["qwen_pass@5_mean"]]

    bars_a = ax.bar([xi - width/2 for xi in x], vals_a, width, label=LABEL_A, color=COLOR_A)
    bars_b = ax.bar([xi + width/2 for xi in x], vals_b, width, label=LABEL_B, color=COLOR_B)

    for bar in list(bars_a) + list(bars_b):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("pass@k (higher is better)", fontsize=11)
    ax.set_title("pass@1 and pass@5 on HumanEval (164 tasks, n=5 samples)", fontsize=11)
    ax.set_ylim(0, min(1.0, max(vals_a + vals_b) * 1.2))
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "pass_at_k.png")


def chart_latency(summary):
    """Bar chart: average generation time."""
    fig, ax = plt.subplots(figsize=(5, 4))
    models = [LABEL_A, LABEL_B]
    times  = [summary["codegen_avg_time"], summary["qwen_avg_time"]]
    colors = [COLOR_A, COLOR_B]

    bars = ax.bar(models, times, color=colors, width=0.4)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.2f}s",
            ha="center", fontsize=10
        )

    ax.set_ylabel("Avg. generation time (seconds)", fontsize=11)
    ax.set_title("Generation Latency per Completion", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "latency.png")


def chart_complexity(summary):
    """Side-by-side: line count and control structures."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, metric, ylabel, title in [
        (axes[0], "avg_lines",    "Avg. lines of code",       "Code Length"),
        (axes[1], "avg_controls", "Avg. control structures",  "Control Flow Complexity"),
    ]:
        vals   = [summary[f"codegen_{metric}"], summary[f"qwen_{metric}"]]
        bars   = ax.bar([LABEL_A, LABEL_B], vals, color=[COLOR_A, COLOR_B], width=0.4)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{bar.get_height():.1f}",
                ha="center", fontsize=10
            )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Code Complexity Metrics", fontsize=12, y=1.02)
    save(fig, "complexity.png")


def chart_rouge(summary):
    """Grouped bar: ROUGE-1 and ROUGE-L for both models."""
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["ROUGE-1", "ROUGE-L"]
    x = range(len(metrics))
    width = 0.3

    vals_a = [summary["codegen_avg_rouge1"], summary["codegen_avg_rougeL"]]
    vals_b = [summary["qwen_avg_rouge1"],    summary["qwen_avg_rougeL"]]

    bars_a = ax.bar([xi - width/2 for xi in x], vals_a, width, label=LABEL_A, color=COLOR_A)
    bars_b = ax.bar([xi + width/2 for xi in x], vals_b, width, label=LABEL_B, color=COLOR_B)

    for bar in list(bars_a) + list(bars_b):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_title("Semantic Similarity to Reference Solutions (ROUGE)\nNote: ROUGE measures text overlap, not correctness", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "rouge_scores.png")


def chart_per_task(df):
    """
    Line plot of pass@1 per task for both models.
    This is the most informative chart — shows exactly where each model struggles.
    Tasks where both models fail tend to require complex reasoning or imports.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df["codegen_pass@1"].values, label=LABEL_A, color=COLOR_A, alpha=0.75, linewidth=1.2)
    ax.plot(df["qwen_pass@1"].values,    label=LABEL_B, color=COLOR_B, alpha=0.75, linewidth=1.2)

    ax.set_xlabel("HumanEval Task Index (0–163)", fontsize=10)
    ax.set_ylabel("pass@1", fontsize=10)
    ax.set_title("per-Task pass@1 Across All 164 HumanEval Tasks", fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "per_task_pass1.png")


if __name__ == "__main__":
    print("Loading results...")
    df, summary = load_data()

    print("Generating charts...")
    chart_pass_at_k(summary)
    chart_latency(summary)
    chart_complexity(summary)
    chart_rouge(summary)
    chart_per_task(df)

    print(f"\nAll charts saved to {OUTPUT_DIR}/")
