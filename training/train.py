"""Fine-tune DistilBERT to classify text into US History TEKS codes.

Uses the same base model and hyperparameters as the Hugging Face AutoTrain run
that produced the published model (connork0/10Batch25Epoch2Grad).

    python training/train.py --output-dir models/teks-distilbert

Point the classifier at the result with TEKS_MODEL=models/teks-distilbert.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "teks_training_data.csv"


def teks_sort_key(code: str) -> tuple[int, str]:
    number, letter = code.split(".")
    return int(number), letter


def load_examples(path: Path) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return [(row["text"], row["target"]) for row in csv.DictReader(file)]


def split_per_label(examples, seed):
    """Hold out one example of every standard, so validation covers every class."""
    by_label = defaultdict(list)
    for text, label in examples:
        by_label[label].append(text)

    rng = random.Random(seed)
    train, validation = [], []
    for label, texts in by_label.items():
        rng.shuffle(texts)
        validation.append((texts[0], label))
        train.extend((text, label) for text in texts[1:])
    return train, validation


def encode(tokenizer, examples, label2id, max_length):
    encodings = tokenizer([text for text, _ in examples], truncation=True, max_length=max_length)
    return [
        {"input_ids": ids, "attention_mask": mask, "labels": label2id[label]}
        for ids, mask, (_, label) in zip(encodings["input_ids"], encodings["attention_mask"], examples)
    ]


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    top3 = np.argsort(logits, axis=-1)[:, ::-1][:, :3]
    return {
        "accuracy": float(np.mean(top3[:, 0] == labels)),
        "top3_accuracy": float(np.mean((top3 == labels[:, None]).any(axis=1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("models/teks-distilbert"))
    parser.add_argument("--base-model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_examples(args.data)
    labels = sorted({label for _, label in examples}, key=teks_sort_key)
    label2id = {label: i for i, label in enumerate(labels)}
    train_examples, validation_examples = split_per_label(examples, args.seed)
    print(f"{len(labels)} standards, {len(train_examples)} training / {len(validation_examples)} validation examples")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label2id,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=encode(tokenizer, train_examples, label2id, args.max_length),
        eval_dataset=encode(tokenizer, validation_examples, label2id, args.max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(trainer.evaluate())

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
