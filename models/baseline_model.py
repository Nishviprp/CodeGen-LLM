# models/baseline_model.py
"""
WHAT THIS FILE DOES:
====================
Wraps Qwen2.5-Coder-1.5B-Instruct as our honest, reproducible baseline.

WHY THIS MODEL INSTEAD OF GITHUB COPILOT?
  GitHub Copilot has no public completions API. Any code that calls
  openai.Completion.create(model="copilot-codex") will fail with a 404 —
  there is no such endpoint. We use a real, open, callable model instead.

WHY QWEN2.5-CODER-1.5B-INSTRUCT?
  - Released Oct 2024 — represents the current state of open-source code LLMs
  - 1.5B params: 4× larger than our fine-tuned model, fair "stronger baseline" framing
  - Instruction-tuned: responds to chat-style prompts, important for code completion
  - Scores ~37% pass@1 on HumanEval out of the box (vs ~12% for CodeGen-350M base)
  - Free to download, no API key needed

HOW INSTRUCTION-TUNED MODELS DIFFER:
  Unlike CodeGen (which just continues text), instruction-tuned models expect a
  system prompt + user message format. We use apply_chat_template() to format
  correctly, then strip any markdown fences from the output.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenCoderBaseline:

    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Qwen] Loading '{model_name}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        print("[Qwen] Ready.\n")

    def generate_code(self, prompt, max_new_tokens=192, temperature=0.6):
        """
        Generate a code completion. Uses chat-template format required by
        instruction-tuned models, then strips any markdown code fences.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Python code completion assistant. "
                    "Complete the given function. Output only the function body — "
                    "no explanation, no markdown, no extra text."
                )
            },
            {
                "role": "user",
                "content": f"Complete this Python function:\n\n{prompt}"
            }
        ]

        # apply_chat_template formats the messages into the model's expected format
        # e.g., "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n..."
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        completion = self.tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True
        )

        # Strip markdown code fences if model added them despite our instructions
        if "```python" in completion:
            completion = completion.split("```python")[1].split("```")[0]
        elif "```" in completion:
            completion = completion.split("```")[1].split("```")[0]

        return completion.strip()
