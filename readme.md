# CodeGen LLM Benchmarking

Fine-tuning and evaluation of an open-source code-generation LLM, benchmarked against industry tools like GitHub Copilot.

## Overview

This project fine-tunes an open-source CodeGen large language model using LoRA (Low-Rank Adaptation via PEFT) on a curated dataset of 5,000+ Python code samples. An automated evaluation framework built on HumanEval is used to benchmark the fine-tuned model's code-generation performance, aiming for results comparable to GitHub Copilot while maintaining full transparency and control over the model.

## Contents

- **main.ipynb** — Core notebook containing the LoRA fine-tuning pipeline and HumanEval-based evaluation framework.
- **Untitled.ipynb** — Supplementary/exploratory notebook.
- **CodeGenReport.pdf** — Written report covering methodology, results, and analysis.

## Approach

- Fine-tune an open-source CodeGen model using LoRA (PEFT) for parameter-efficient training.
- Train on 5,000+ Python code samples.
- Evaluate generated code using the HumanEval benchmark.
- Compare results against existing tools (e.g., GitHub Copilot) on functional correctness.

## Requirements

- Python 3.11+
- Hugging Face `transformers` and `peft`
- PyTorch
- HumanEval evaluation harness

Install dependencies:

```bash
pip install transformers peft torch
```

## Usage

Open `main.ipynb` in Jupyter and run the cells sequentially to reproduce the fine-tuning and evaluation pipeline.

```bash
jupyter notebook main.ipynb
```

## Report

See `CodeGenReport.pdf` for full methodology, benchmark results, and discussion.
