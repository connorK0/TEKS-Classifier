"""Wraps the fine-tuned DistilBERT model that maps lesson text to TEKS codes."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TeksClassifier:
    def __init__(self, model_id: str, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        # The model was fine-tuned on 128-token inputs, so predict on the same length.
        self.max_length = max_length

    @property
    def labels(self) -> set[str]:
        return set(self.model.config.id2label.values())

    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the k most likely TEKS codes with their probabilities, best first."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        with torch.no_grad():
            probabilities = self.model(**inputs).logits.softmax(dim=-1)[0]
        top = probabilities.topk(k)
        id2label = self.model.config.id2label
        return [(id2label[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
