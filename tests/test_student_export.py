"""Export a tiny random student to ONNX and run it through the engine's OnnxAnalyzer.

Guards the Colab -> Docker hand-off: if this passes, a model trained in the notebook loads in the engine.
Skipped when the training extras (torch, transformers, onnx) are not installed.
"""

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from ripple.engine.analyzers import OnnxAnalyzer  # noqa: E402
from ripple.schemas import EventType  # noqa: E402
from ripple.training.student import Student, export_onnx  # noqa: E402

TEXTS = ["Facebook shares plunge after Q2 revenue miss", "Trade war fears grow", "ok"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A 2-layer BERT with the real bge-small vocabulary layout, randomly initialised (no download)."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({w.lower() for t in TEXTS for w in t.split()})
    vocab_file = tmp_path_factory.mktemp("tok") / "vocab.txt"
    vocab_file.write_text("\n".join(vocab))
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab_file))
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64
    )
    torch.manual_seed(0)
    return Student(transformers.BertModel(config)).eval(), tokenizer


def test_onnx_matches_torch_and_loads_in_engine(tiny, tmp_path):
    student, tokenizer = tiny
    export_onnx(student, tokenizer, tmp_path, max_length=32, quantize=False)
    analyses = OnnxAnalyzer(tmp_path).analyze(TEXTS)

    batch = tokenizer(TEXTS, padding=True, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        event, sentiment, severity = student(batch["input_ids"], batch["attention_mask"])
    p = sentiment.softmax(-1)
    expected_sentiment = (p[:, 2] - p[:, 0]).tolist()

    assert len(analyses) == len(TEXTS)
    for a, ev, s, sev in zip(analyses, event.argmax(-1).tolist(), expected_sentiment, severity.sigmoid().tolist()):
        assert isinstance(a.event.event_type, EventType)
        assert list(EventType)[ev] == a.event.event_type
        assert a.sentiment == pytest.approx(s, abs=1e-3)
        assert a.severity == pytest.approx(sev, abs=1e-3)


def test_int8_export_runs(tiny, tmp_path):
    student, tokenizer = tiny
    export_onnx(student, tokenizer, tmp_path, max_length=32)
    analyses = OnnxAnalyzer(tmp_path).analyze(TEXTS)
    assert all(-1 <= a.sentiment <= 1 and 0 <= a.event.confidence <= 1 for a in analyses)
