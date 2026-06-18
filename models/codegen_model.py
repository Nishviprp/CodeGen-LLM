# models/codegen_model.py
"""
WHAT THIS FILE DOES:
====================
Wraps the fine-tuned CodeGen-350M model so the evaluator can call it cleanly.

HOW LOADING WORKS:
  1. Load the base model (Salesforce/codegen-350M-mono) from Hugging Face Hub
  2. Load the LoRA adapter from models/finetuned-adapter/ (the ~4MB file we trained)
  3. PeftModel merges them — the adapter matrices modify attention weights on the fly

HOW GENERATION WORKS:
  Input:  "def has_close_elements(numbers, threshold):\n"   ← the HumanEval prompt
  Output: "    for i in range(len(numbers)):\n    ..."      ← the generated completion

  The model only returns the NEW tokens (the completion), not the prompt itself.
  temperature=0.6: controls randomness. Lower = more deterministic. 0.6 is standard
  for code generation (not too greedy, not too random).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class FineTunedCodeGen:

    def __init__(self, base_model="Salesforce/codegen-350M-mono",
                 adapter_path="models/finetuned-adapter"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CodeGen] Loading base model '{base_model}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        print(f"[CodeGen] Loading LoRA adapter from '{adapter_path}'...")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        print("[CodeGen] Ready.\n")

    def generate_code(self, prompt, max_new_tokens=192, temperature=0.6):
        """
        Generate a code completion for the given prompt.
        Returns only the generated part (excludes the input prompt).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,            # sample from distribution (not greedy)
                temperature=temperature,
                top_p=0.95,               # nucleus sampling
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (slice off the input)
        new_tokens = output[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
