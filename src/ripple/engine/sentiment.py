"""Sentiment scoring with FinBERT (ProsusAI/finbert).

score = P(positive) - P(negative), giving a value in [-1, 1].
"""

import os

SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")


class SentimentModel:
    def __init__(self, model_name: str = SENTIMENT_MODEL, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self._pos, self._neg = labels["positive"], labels["negative"]
        self.batch_size = batch_size

    def score(self, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[i : i + self.batch_size], padding=True, truncation=True, max_length=96, return_tensors="pt"
            )
            with self._torch.inference_mode():
                probs = self.model(**batch).logits.softmax(dim=-1)
            scores.extend((probs[:, self._pos] - probs[:, self._neg]).tolist())
        return [round(s, 4) for s in scores]
