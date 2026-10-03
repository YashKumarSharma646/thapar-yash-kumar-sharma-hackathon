"""Text analyzers: text -> (sentiment, event). The engine works with either backend.

- RuleAnalyzer:  keyword event rules + a sentiment scorer (FinBERT). The Day 2 baseline.
- OnnxAnalyzer:  one distilled multi-task transformer (event + sentiment + severity), trained on
                 LLM-teacher labels (notebooks/04_label_and_distill.ipynb) and run with ONNX Runtime.
                 No PyTorch needed at inference.

Both report the rule phrases that support the chosen event, so the evidence card stays readable.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ripple.config import MODELS_DIR
from ripple.engine.events import EventResult, classify_event, phrases_for
from ripple.schemas import EventType

log = logging.getLogger(__name__)

DISTILLED_DIR = MODELS_DIR / "ripple-nlp"
NLP_BACKEND = os.getenv("NLP_BACKEND", "auto")  # auto | onnx | rules


@dataclass
class Analysis:
    sentiment: float
    event: EventResult
    severity: float | None = None  # model's 0-1 materiality estimate, when the backend provides one


class Analyzer(Protocol):
    name: str

    def analyze(self, texts: list[str]) -> list[Analysis]: ...


class Scorer(Protocol):
    def score(self, texts: list[str]) -> list[float]: ...


class RuleAnalyzer:
    name = "rules+finbert"

    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    def analyze(self, texts: list[str]) -> list[Analysis]:
        sentiments = self.scorer.score(texts) if texts else []
        return [Analysis(s, classify_event(t)) for t, s in zip(texts, sentiments)]


class OnnxAnalyzer:
    name = "distilled-onnx"

    def __init__(self, model_dir: Path = DISTILLED_DIR, batch_size: int = 64):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        cfg = json.loads((model_dir / "config.json").read_text())
        self.events = [EventType(e) for e in cfg["event_labels"]]
        sentiment_labels = cfg["sentiment_labels"]
        self._pos, self._neg = sentiment_labels.index("positive"), sentiment_labels.index("negative")

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(cfg["max_length"])
        self.tokenizer.enable_padding(pad_id=cfg["pad_token_id"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._np = np
        self.batch_size = batch_size

    def analyze(self, texts: list[str]) -> list[Analysis]:
        np, out = self._np, []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer.encode_batch(texts[i : i + self.batch_size])
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            event_p, sent_p, severity = self.session.run(["event_probs", "sentiment_probs", "severity"], feeds)
            for text, ep, sp, sev in zip(texts[i : i + self.batch_size], event_p, sent_p, severity):
                event = self.events[int(ep.argmax())]
                out.append(
                    Analysis(
                        sentiment=round(float(sp[self._pos] - sp[self._neg]), 4),
                        event=EventResult(event, round(float(ep.max()), 3), phrases_for(text, event)),
                        severity=round(float(sev), 3),
                    )
                )
        return out


def load_analyzer(backend: str = NLP_BACKEND) -> Analyzer:
    has_model = (DISTILLED_DIR / "model.onnx").exists()
    if backend == "onnx" or (backend == "auto" and has_model):
        log.info("NLP backend: distilled ONNX model (%s)", DISTILLED_DIR)
        return OnnxAnalyzer()
    if backend == "auto":
        log.info("No distilled model at %s, using rules + FinBERT", DISTILLED_DIR)
    from ripple.engine.sentiment import SentimentModel

    return RuleAnalyzer(SentimentModel())
