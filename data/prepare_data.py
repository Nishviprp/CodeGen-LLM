# data/prepare_data.py
"""
WHAT THIS FILE DOES:
====================
Downloads and prepares two datasets:

1. TRAINING DATA (CodeParrot):
   - Source: huggingface.co/datasets/codeparrot/codeparrot-clean
   - What it is: ~180GB of Python code scraped from GitHub, pre-cleaned
   - We take 5000 samples (enough to fine-tune a small model efficiently)
   - We split each file into a "prompt" (first 1/3 of lines) and
     "completion" (remaining 2/3) to create supervised training pairs
   - Saved to: data/python_dataset.json

2. EVALUATION DATA (HumanEval):
   - Source: huggingface.co/datasets/openai_humaneval
   - What it is: 164 hand-written Python programming tasks from OpenAI
   - Each task has: a function prompt, a reference solution, and unit tests
   - This is the STANDARD benchmark for evaluating code generation models
   - Saved to: data/human_eval.json

WHY THESE DATASETS?
   CodeParrot  → real Python code, diverse, permissively licensed
   HumanEval   → reproducible, standardized, used in every major paper
                 (Codex, StarCoder, CodeLlama, DeepSeek-Coder all use it)
"""

import os
import json
from datasets import load_dataset


def prepare_training_data(num_examples=5000, output_path="data/python_dataset.json"):
    """
    Downloads CodeParrot, extracts prompt/completion pairs, saves to JSON.

    Each resulting item looks like:
        {
            "prompt":     "import os\nimport sys\n",          # first ~1/3 of file
            "completion": "def main():\n    ...\n"            # remaining ~2/3
        }
    The model learns: given the start of a file, predict the rest.
    """
    print(f"Downloading CodeParrot dataset ({num_examples} examples)...")
    print("This may take a few minutes on first run (downloads ~500MB).\n")

    # Load from Hugging Face Hub — streaming=True avoids downloading the full 180GB
    dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True       # <-- KEY: we stream and take only what we need
    )

    formatted_data = []
    skipped = 0

    for i, item in enumerate(dataset):
        if len(formatted_data) >= num_examples:
            break

        code = item["content"]   # the raw Python file as a string
        lines = code.split("\n")

        # Skip files that are too short to split meaningfully
        if len(lines) < 6:
            skipped += 1
            continue

        # Split into prompt (first third) and completion (rest)
        split_point = max(3, len(lines) // 3)
        prompt = "\n".join(lines[:split_point])
        completion = "\n".join(lines[split_point:])

        formatted_data.append({
            "prompt": prompt,
            "completion": completion
        })

        if len(formatted_data) % 500 == 0:
            print(f"  Collected {len(formatted_data)}/{num_examples} examples...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(formatted_data, f)

    print(f"\nDone. Saved {len(formatted_data)} training examples to {output_path}")
    print(f"(Skipped {skipped} files that were too short)\n")
    return output_path


def prepare_eval_data(output_path="data/human_eval.json"):
    """
    Downloads HumanEval and saves it as a flat JSON list.

    Each resulting item looks like:
        {
            "task_id":          "HumanEval/0",
            "prompt":           "def has_close_elements(numbers, threshold):\n    ...",
            "canonical_solution": "    for idx, elem in enumerate(numbers):\n    ...",
            "test":             "def check(candidate):\n    assert ...",
            "entry_point":      "has_close_elements"
        }

    During evaluation:
        - We feed `prompt` to the model
        - Model generates the function body
        - We run: prompt + generated_body + test_code
        - If check(entry_point) passes all assertions → success
    """
    print("Downloading HumanEval benchmark (164 tasks)...")

    dataset = load_dataset("openai_humaneval", split="test")

    data = []
    for item in dataset:
        data.append({
            "task_id":            item["task_id"],
            "prompt":             item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test":               item["test"],
            "entry_point":        item["entry_point"]
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Done. Saved {len(data)} HumanEval tasks to {output_path}\n")
    return output_path


if __name__ == "__main__":
    prepare_training_data(
        num_examples=5000,
        output_path="data/python_dataset.json"
    )
    prepare_eval_data(
        output_path="data/human_eval.json"
    )
    print("All data ready. Next step: python models/finetune.py")
