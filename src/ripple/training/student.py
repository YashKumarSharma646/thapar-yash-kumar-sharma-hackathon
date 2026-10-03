"""Distilled multi-task student: one small encoder, three heads (event, sentiment, severity).

Trained on Colab from LLM-teacher labels (notebooks/04_label_and_distill.ipynb), then exported to
int8 ONNX for CPU inference by ripple.engine.analyzers.OnnxAnalyzer. Lives in the package so the
notebook and the local export smoke test (tests/test_student_export.py) share the exact same code.
Needs torch + transformers + onnx + onnxruntime (training only; the engine needs just onnxruntime).
"""

import json
from pathlib import Path

import torch
from torch import nn

from ripple.schemas import EventType

EVENT_LABELS = [e.value for e in EventType]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
OUTPUT_NAMES = ["event_probs", "sentiment_probs", "severity"]


class Student(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.event_head = nn.Linear(hidden, len(EVENT_LABELS))
        self.sentiment_head = nn.Linear(hidden, len(SENTIMENT_LABELS))
        self.severity_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.dropout(cls)
        return self.event_head(x), self.sentiment_head(x), self.severity_head(x).squeeze(-1)


class _InferenceWrapper(nn.Module):
    """Probabilities out, so the runtime needs no post-processing beyond argmax."""

    def __init__(self, student: Student):
        super().__init__()
        self.student = student

    def forward(self, input_ids, attention_mask):
        event, sentiment, severity = self.student(input_ids, attention_mask)
        return event.softmax(-1), sentiment.softmax(-1), severity.sigmoid()


def export_onnx(
    student: Student, tokenizer, out_dir: Path, max_length: int, extra_config: dict | None = None, quantize: bool = True
) -> Path:
    """Write model.onnx (int8 by default), tokenizer.json and config.json to out_dir."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = _InferenceWrapper(student).eval().cpu()
    dummy = tokenizer(["export example", "a second, longer export example"], padding=True, return_tensors="pt")
    args = (dummy["input_ids"], dummy["attention_mask"])
    fp32 = out_dir / "model_fp32.onnx"
    axes = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}}
    kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=OUTPUT_NAMES,
        dynamic_axes=axes | {name: {0: "batch"} for name in OUTPUT_NAMES},
        opset_version=17,
    )
    with torch.no_grad():
        try:  # the TorchScript exporter is the most reliable for BERT-style encoders...
            torch.onnx.export(wrapper, args, str(fp32), dynamo=False, **kwargs)
        except Exception:  # ...but newer PyTorch releases may drop it
            torch.onnx.export(wrapper, args, str(fp32), dynamo=True, external_data=False, **kwargs)
    final = out_dir / "model.onnx"
    if quantize:
        quantize_dynamic(str(fp32), str(final), weight_type=QuantType.QInt8)
        fp32.unlink()
    else:
        fp32.replace(final)

    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    config = {
        "event_labels": EVENT_LABELS,
        "sentiment_labels": SENTIMENT_LABELS,
        "max_length": max_length,
        "pad_token_id": tokenizer.pad_token_id,
    } | (extra_config or {})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return final
