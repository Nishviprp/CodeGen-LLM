"""
reextract_experiment.py  (v2 — corrected normalizer)

Quantifies the output-format confound between a base-completion model (CodeGen,
emits an indented body to continue the prompt) and an instruction-tuned model
(Qwen, answers at column 0 / as a full function). The original harness assumes the
base-completion format, structurally penalizing the instruct model's outputs.

LIMITATION: detailed_results.csv stored only completion #1 of 5 per task. Reported
numbers are a SINGLE-SAMPLE pass@1 proxy; the naive->normalized DELTA on the
identical completion set is a conservative lower bound on the confound. Regenerate
all 5 completions (Colab/GPU/HF) for the true pass@1/pass@5.

Schemes (both append: + test + check(entry_point)):
  NAIVE      : prompt + raw_completion                       (original harness)
  NORMALIZED : format-aware repair, applied identically to BOTH models:
                 - if completion defines `def {entry_point}`: run it standalone,
                   prepending only the prompt's import header (no redeclared def);
                 - else (bare body): dedent, then re-indent 4 spaces so it nests
                   under the prompt's def header.
Applying NORMALIZED to both models is the symmetry control: it should barely move
CodeGen (whose native format already matches naive) while recovering Qwen.
"""
import ast, json, re, subprocess, tempfile, textwrap
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = {t["task_id"]: t for t in json.load(open(ROOT/"data"/"human_eval.json"))}
DF = pd.read_csv(ROOT/"results"/"detailed_results.csv")
TIMEOUT = 5

def run_program(src):
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(src); tmp = f.name
    try:
        r = subprocess.run(["python", tmp], capture_output=True, text=True, timeout=TIMEOUT)
        if r.returncode == 0: return "OK"
        e = r.stderr[-300:]
        if "SyntaxError" in e or "IndentationError" in e: return "SYNTAX"
        if "AssertionError" in e: return "ASSERTION"
        return "RUNTIME"
    except subprocess.TimeoutExpired: return "TIMEOUT"
    except Exception: return "RUNTIME"

def import_header(prompt, ep):
    out = []
    for line in prompt.splitlines():
        if re.match(rf"\s*def\s+{re.escape(ep)}\b", line): break
        out.append(line)
    return "\n".join(out)

def strip_fences(c):
    if "```" in c:
        m = re.search(r"```(?:python)?\s*(.*?)```", c, re.DOTALL)
        if m: return m.group(1)
    return c

def normalize(completion, prompt, ep):
    c = strip_fences(completion)
    if re.search(rf"\bdef\s+{re.escape(ep)}\b", c):           # full function -> standalone
        lines = c.splitlines()
        for i, ln in enumerate(lines):
            if re.match(r"\s*(from |import |def )", ln):
                c = "\n".join(lines[i:]); break
        return import_header(prompt, ep) + "\n" + c
    body = textwrap.indent(textwrap.dedent(c), "    ")        # bare body -> nest under def
    return prompt + body

def score(col, scheme):
    npass = 0; cats = {k:0 for k in ("OK","SYNTAX","ASSERTION","RUNTIME","TIMEOUT")}
    for _, row in DF.iterrows():
        t = TASKS[row["task_id"]]; comp = row[col]
        if not isinstance(comp, str): continue
        tail = "\n\n" + t["test"] + f"\n\ncheck({t['entry_point']})\n"
        src = (t["prompt"] + comp + tail) if scheme == "naive" else (normalize(comp, t["prompt"], t["entry_point"]) + tail)
        res = run_program(src); cats[res] += 1
        if res == "OK": npass += 1
    return npass, cats

def parse_checks(col):
    """Symmetric validity via compile() (catches 'return outside function', unlike ast.parse)."""
    stand = ctx = n = 0
    for _, row in DF.iterrows():
        comp = row[col]
        if not isinstance(comp, str): continue
        t = TASKS[row["task_id"]]; n += 1
        try: compile(comp, "<s>", "exec"); stand += 1
        except Exception: pass
        try: compile(t["prompt"] + comp, "<s>", "exec"); ctx += 1
        except Exception: pass
    return stand/n*100, ctx/n*100

if __name__ == "__main__":
    n = len(DF)
    print(f"Scoring {n} stored completion-1 outputs/model (single-sample pass@1 proxy)\n")
    qn, qnc = score("qwen_sample", "naive")
    qN, qNc = score("qwen_sample", "normalized")
    cn, cnc = score("codegen_sample", "naive")
    cN, cNc = score("codegen_sample", "normalized")
    print("QWEN2.5-Coder-1.5B-Instruct")
    print(f"  naive      {qn:3d}/{n} = {qn/n*100:5.1f}%   {qnc}")
    print(f"  normalized {qN:3d}/{n} = {qN/n*100:5.1f}%   {qNc}")
    print(f"  >> DELTA +{(qN-qn)/n*100:.1f} pts ({qn/n*100:.1f}% -> {qN/n*100:.1f}%)\n")
    print("CodeGen-350M (fine-tuned)  [symmetry control]")
    print(f"  naive      {cn:3d}/{n} = {cn/n*100:5.1f}%   {cnc}")
    print(f"  normalized {cN:3d}/{n} = {cN/n*100:5.1f}%   {cNc}")
    print(f"  >> DELTA +{(cN-cn)/n*100:.1f} pts\n")
    qs, qc = parse_checks("qwen_sample"); cs, cc = parse_checks("codegen_sample")
    print("Symmetric validity (compile succeeds?)")
    print(f"  {'model':<26}{'standalone':>12}{'in-context':>12}")
    print(f"  {'Qwen (col-0 answers)':<26}{qs:>11.1f}%{qc:>11.1f}%")
    print(f"  {'CodeGen (indented body)':<26}{cs:>11.1f}%{cc:>11.1f}%")
    pd.DataFrame([
        {"model":"qwen","scheme":"naive","pass":qn,"n":n,"pass_rate_pct":round(qn/n*100,1),**qnc},
        {"model":"qwen","scheme":"normalized","pass":qN,"n":n,"pass_rate_pct":round(qN/n*100,1),**qNc},
        {"model":"codegen","scheme":"naive","pass":cn,"n":n,"pass_rate_pct":round(cn/n*100,1),**cnc},
        {"model":"codegen","scheme":"normalized","pass":cN,"n":n,"pass_rate_pct":round(cN/n*100,1),**cNc},
    ]).to_csv(ROOT/"results"/"reextract_results.csv", index=False)
    print("\nWrote results/reextract_results.csv")
