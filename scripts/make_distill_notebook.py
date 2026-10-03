"""Generates notebooks/04_label_and_distill.ipynb. Usage: python scripts/make_distill_notebook.py notebooks/04_label_and_distill.ipynb"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Ripple · Day 4: LLM-teacher labelling and distillation

**Runtime → Change runtime type → L4 GPU**, then **Runtime → Run all**. It takes about 30–45 minutes and needs no input.
At the end, the browser downloads **`ripple-nlp.zip`**. Put it in the project folder.

What it does:
1. **Teacher:** `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) labels ~21k headlines and tweets with *event type*, *sentiment* and *severity (1–5)*.
2. **Student:** `BAAI/bge-small-en-v1.5` (33M params, MIT) is fine-tuned on those labels with three heads.
3. **Evaluation:** on the held-out replay window (15 Jul – 31 Aug 2018, the demo data, never trained on), against the Day 2 baselines (keyword rules, FinBERT).
4. **Export:** int8 ONNX (~35 MB) for fast CPU inference in Docker, with no PyTorch needed at runtime.

If the session crashes with *CUDA out of memory* in the training step: **Runtime → Restart session**, then **Run all** again. Finished teacher labels are cached and not recomputed.
""")

code("""
# 1. Setup: GPU check, project code + corpus, libraries
!nvidia-smi --query-gpu=name,memory.total --format=csv
REPO = "https://github.com/YashKumarSharma646/thapar-yash-kumar-sharma-hackathon.git"
!test -d /content/repo || git clone -q --depth 1 {REPO} /content/repo
!pip install -q vllm onnx onnxruntime 2>&1 | tail -2

import sys
sys.path.insert(0, "/content/repo/src")
""")

code("""
import gc, json, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch

TEACHER = "Qwen/Qwen2.5-7B-Instruct"
STUDENT = "BAAI/bge-small-en-v1.5"
MAX_LEN = 96
WORK = Path("/content/work"); WORK.mkdir(exist_ok=True)
LABELS_CSV = WORK / "teacher_labels.csv"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from ripple.training.student import EVENT_LABELS, SENTIMENT_LABELS
corpus = pd.read_csv("/content/repo/data/labeling/corpus.csv")
print(corpus.groupby(["split", "source"]).size(), "\\nGPU:", torch.cuda.get_device_name(0) if DEV == "cuda" else "NONE - switch the runtime to L4 GPU!")
""")

md("## 2. Teacher labelling")

code('''
SYSTEM = """You label financial news headlines and stock-market tweets for a bank's risk engine.
Return ONLY a JSON object: {"event": <event>, "sentiment": <sentiment>, "severity": <1-5>}

event: the main type of event the text reports (pick exactly one):
- "Earnings": quarterly results, revenue, profit, EPS, guidance, earnings previews, dividends.
- "Credit Event": default, bankruptcy, credit-rating actions by rating agencies, debt restructuring, missed payments, liquidity crises.
- "Geopolitical": trade wars, tariffs, sanctions, wars, international conflicts, political instability.
- "Macroeconomic": central banks, interest rates, inflation, GDP, jobs data, currencies, oil/commodity prices, the economy.
- "Merger/Acquisition": acquisitions, mergers, takeovers, stake purchases, divestitures, spin-offs.
- "Product Launch": new products, services or features being launched or unveiled.
- "Regulatory/Legal": lawsuits, regulators, investigations, fines, legislation, privacy rules, drug/product approvals.
- "Operational": data breaches, cyberattacks, outages, recalls, accidents, strikes, layoffs, executive departures, supply disruptions.
- "Other": none of the above, including analyst rating or price-target changes, stock moves with no stated cause, market chatter, listicles, opinion.

sentiment: the text's implication for the company's (or, if none is named, the market's) investors and creditors: "negative", "neutral" or "positive".

severity: how material the news is for the company's (or market's) financial or credit risk:
1 = routine / negligible, 2 = minor, 3 = moderate, 4 = major, 5 = severe (default, record loss, crisis, war).

Examples:
"Facebook shares plunge 19% after Q2 revenue miss and weak outlook" -> {"event": "Earnings", "sentiment": "negative", "severity": 5}
"Goldman downgrades Rockwell Automation to neutral" -> {"event": "Other", "sentiment": "negative", "severity": 2}
"Moody's cuts Turkey's sovereign rating deeper into junk as lira plunges" -> {"event": "Credit Event", "sentiment": "negative", "severity": 5}
"US to impose 25% tariffs on $16 billion of Chinese goods" -> {"event": "Geopolitical", "sentiment": "negative", "severity": 4}
"Apple unveils three new iPhones" -> {"event": "Product Launch", "sentiment": "positive", "severity": 2}
"""

EVENT_BY_KEY = {re.sub(r"\\W", "", e.lower()): e for e in EVENT_LABELS}
JSON_RE = re.compile(r"\\{.*?\\}", re.S)

def parse(raw: str):
    m = JSON_RE.search(raw or "")
    try:
        d = json.loads(m.group(0))
        event = EVENT_BY_KEY[re.sub(r"\\W", "", str(d["event"]).lower())]
        sentiment = str(d["sentiment"]).lower().strip()
        severity = int(d["severity"])
        assert sentiment in SENTIMENT_LABELS and 1 <= severity <= 5
        return event, sentiment, severity
    except Exception:
        return None, None, None
''')

code('''
def label_with_vllm(texts):
    from vllm import LLM, SamplingParams
    llm = LLM(TEACHER, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.80, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=48))
    return [o.outputs[0].text for o in outs], llm

def label_with_transformers(texts, batch_size=48):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    raws = []
    for i in range(0, len(texts), batch_size):
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                                           tokenize=False, add_generation_prompt=True) for t in texts[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=48, do_sample=False)
        raws += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        if i % (batch_size * 20) == 0: print(f"{i:,}/{len(texts):,}")
    return raws, model

if LABELS_CSV.exists():
    labels = pd.read_csv(LABELS_CSV)
    print("Using cached teacher labels:", len(labels))
else:
    t0 = time.time()
    texts = corpus.text.tolist()
    try:
        raws, teacher = label_with_vllm(texts)
    except Exception as e:
        print("vLLM unavailable, falling back to transformers (slower):", repr(e)[:300])
        raws, teacher = label_with_transformers(texts)
    print(f"labelled {len(texts):,} texts in {(time.time() - t0) / 60:.1f} min")
    parsed = [parse(r) for r in raws]
    labels = corpus.assign(teacher_event=[p[0] for p in parsed], teacher_sentiment=[p[1] for p in parsed],
                           teacher_severity=[p[2] for p in parsed], teacher_raw=raws)
    labels.to_csv(LABELS_CSV, index=False)
    del teacher; gc.collect(); torch.cuda.empty_cache()

print(f"unparseable: {labels.teacher_event.isna().mean():.2%}")
labels = labels.dropna(subset=["teacher_event"]).copy()
labels["teacher_severity"] = labels.teacher_severity.astype(int)
print(labels.teacher_event.value_counts(), labels.teacher_sentiment.value_counts(), labels.teacher_severity.value_counts().sort_index(), sep="\\n\\n")
''')

code('''
# Spot-check: a few teacher labels per event class (read these!)
pd.set_option("display.max_colwidth", 110)
for ev in EVENT_LABELS:
    s = labels[labels.teacher_event == ev].sample(min(3, (labels.teacher_event == ev).sum()), random_state=1)
    display(s[["text", "teacher_event", "teacher_sentiment", "teacher_severity"]])
''')

md("## 3. Train the student (bge-small, three heads)")

code('''
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from ripple.training.student import Student

torch.manual_seed(0); np.random.seed(0)
ev_idx = {e: i for i, e in enumerate(EVENT_LABELS)}; se_idx = {s: i for i, s in enumerate(SENTIMENT_LABELS)}
train_all = labels[labels.split == "train"]
train_df, dev_df = train_test_split(train_all, test_size=0.1, random_state=0, stratify=train_all.teacher_event)
eval_df = labels[labels.split == "eval"]
print(len(train_df), len(dev_df), len(eval_df))

tokenizer = AutoTokenizer.from_pretrained(STUDENT)

def tensors(df):
    enc = tokenizer(df.text.tolist(), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    return (enc.input_ids, enc.attention_mask, torch.tensor(df.teacher_event.map(ev_idx).values),
            torch.tensor(df.teacher_sentiment.map(se_idx).values),
            torch.tensor((df.teacher_severity.values - 1) / 4.0, dtype=torch.float))

def loader(df, shuffle, bs=64):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors(df)), batch_size=bs, shuffle=shuffle)

@torch.no_grad()
def predict(model, df, bs=256):
    model.eval(); ev, se, sv = [], [], []
    for ids, mask, *_ in loader(df, False, bs):
        with torch.autocast(DEV, dtype=torch.bfloat16):
            e, s, v = model(ids.to(DEV), mask.to(DEV))
        ev.append(e.float().argmax(-1).cpu()); se.append(s.float().argmax(-1).cpu()); sv.append(v.float().sigmoid().cpu())
    return torch.cat(ev).numpy(), torch.cat(se).numpy(), torch.cat(sv).numpy()

def scores(model, df):
    ev, se, sv = predict(model, df)
    y_ev, y_se = df.teacher_event.map(ev_idx).values, df.teacher_sentiment.map(se_idx).values
    return {"event_macro_f1": f1_score(y_ev, ev, average="macro"), "event_acc": accuracy_score(y_ev, ev),
            "sentiment_macro_f1": f1_score(y_se, se, average="macro"), "sentiment_acc": accuracy_score(y_se, se)}

# Rare classes get more weight (inverse square-root frequency).
counts = train_df.teacher_event.map(ev_idx).value_counts().reindex(range(len(EVENT_LABELS)), fill_value=1).values
event_weights = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float); event_weights /= event_weights.mean()

model = Student(AutoModel.from_pretrained(STUDENT)).to(DEV)
EPOCHS, LR = 5, 8e-5
train_loader = loader(train_df, True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * EPOCHS * len(train_loader)), EPOCHS * len(train_loader))
ce_ev = torch.nn.CrossEntropyLoss(weight=event_weights.to(DEV)); ce_se = torch.nn.CrossEntropyLoss()

best, best_state = -1, None
for epoch in range(EPOCHS):
    model.train(); total = 0
    for ids, mask, y_ev, y_se, y_sv in train_loader:
        ids, mask, y_ev, y_se, y_sv = (t.to(DEV) for t in (ids, mask, y_ev, y_se, y_sv))
        with torch.autocast(DEV, dtype=torch.bfloat16):
            e, s, v = model(ids, mask)
        loss = ce_ev(e.float(), y_ev) + ce_se(s.float(), y_se) + 2.0 * torch.nn.functional.mse_loss(v.float().sigmoid(), y_sv)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        total += loss.item()
    dev = scores(model, dev_df)
    print(f"epoch {epoch + 1}: loss {total / len(train_loader):.3f}  " + "  ".join(f"{k} {v:.3f}" for k, v in dev.items()))
    if dev["event_macro_f1"] + dev["sentiment_macro_f1"] > best:
        best = dev["event_macro_f1"] + dev["sentiment_macro_f1"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
''')

md("""
## 4. Evaluation on the held-out replay window (the demo data)

The reference is the teacher's labels. The question is how much of the 7B teacher's judgement the 33M student keeps, and how both compare with the Day 2 baselines.
""")

code('''
from scipy.stats import spearmanr
ev, se, sv = predict(model, eval_df)
y_ev = eval_df.teacher_event.map(ev_idx).values; y_se = eval_df.teacher_sentiment.map(se_idx).values

# Baseline 1: keyword rules (Day 2), already in the corpus.
rule_ev = eval_df.rule_event.map(ev_idx).values
# Baseline 2: FinBERT sentiment (Day 2).
from transformers import pipeline
finbert = pipeline("text-classification", model="ProsusAI/finbert", device=0 if DEV == "cuda" else -1, truncation=True, max_length=MAX_LEN)
fb_se = np.array([se_idx[p["label"].lower()] for p in finbert(eval_df.text.tolist(), batch_size=128)])
del finbert; gc.collect(); torch.cuda.empty_cache()

metrics = {
    "eval_docs": int(len(eval_df)),
    "event_macro_f1": {"rules": f1_score(y_ev, rule_ev, average="macro"), "student": f1_score(y_ev, ev, average="macro")},
    "event_accuracy": {"rules": accuracy_score(y_ev, rule_ev), "student": accuracy_score(y_ev, ev)},
    "sentiment_macro_f1": {"finbert": f1_score(y_se, fb_se, average="macro"), "student": f1_score(y_se, se, average="macro")},
    "sentiment_accuracy": {"finbert": accuracy_score(y_se, fb_se), "student": accuracy_score(y_se, se)},
    "severity_spearman": {"student": spearmanr(eval_df.teacher_severity, sv).correlation},
}
metrics = json.loads(json.dumps(metrics, default=float), parse_float=lambda x: round(float(x), 4))
print(json.dumps(metrics, indent=2))
print("\\nStudent event report (vs teacher):")
print(classification_report(y_ev, ev, labels=range(len(EVENT_LABELS)), target_names=EVENT_LABELS, digits=3, zero_division=0))
print("Rules event report (vs teacher):")
print(classification_report(y_ev, rule_ev, labels=range(len(EVENT_LABELS)), target_names=EVENT_LABELS, digits=3, zero_division=0))
''')

md("## 5. Export to int8 ONNX and verify it against PyTorch")

code('''
import onnxruntime as ort
from ripple.engine.analyzers import OnnxAnalyzer
from ripple.training.student import export_onnx

OUT = WORK / "ripple-nlp"
model = model.float().cpu().eval()
export_onnx(model, tokenizer, OUT, MAX_LEN, extra_config={
    "backbone": STUDENT, "teacher": TEACHER, "train_docs": int(len(train_df)), "metrics": metrics})
print({p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in OUT.iterdir()})

analyzer = OnnxAnalyzer(OUT)
sample = eval_df.sample(min(2000, len(eval_df)), random_state=0)
t0 = time.time(); onnx_out = analyzer.analyze(sample.text.tolist()); dt = time.time() - t0
model.to(DEV)
ev_t, se_t, _ = predict(model, sample)
agree_ev = np.mean([a.event.event_type.value == EVENT_LABELS[i] for a, i in zip(onnx_out, ev_t)])
print(f"int8 ONNX vs PyTorch agreement on events: {agree_ev:.1%}")
print(f"CPU throughput (Colab, {__import__('os').cpu_count()} vCPU): {len(sample) / dt:.0f} docs/s")
metrics["onnx_int8_event_agreement"] = round(float(agree_ev), 4)
cfg = json.loads((OUT / "config.json").read_text()); cfg["metrics"] = metrics
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))
''')

md("## 6. Package and download")

code('''
import shutil
labels.drop(columns=["teacher_raw"]).to_csv(OUT / "teacher_labels.csv", index=False)
shutil.make_archive("/content/ripple-nlp", "zip", OUT)
print(f"ripple-nlp.zip: {Path('/content/ripple-nlp.zip').stat().st_size / 1e6:.1f} MB")
from google.colab import files
files.download("/content/ripple-nlp.zip")
''')

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False))
print("cells:", len(cells))
