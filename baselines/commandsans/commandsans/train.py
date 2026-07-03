"""Train the CommandSans token classifier.

Recipe (following the CommandSans paper): XLM-RoBERTa token classification
with binary word labels (INSTRUCTION/DATA), 512-token windows with 256
overlap, class-weighted cross-entropy against label imbalance, splits made at
the example level.

    python -m commandsans.train --train-data data/labeled.jsonl \
        --output-dir checkpoints/commandsans
"""

from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from commandsans.data import IGNORE_INDEX, WindowedTokenDataset, collate, load_labeled_jsonl


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CommandSans instruction classifier")
    parser.add_argument("--train-data", required=True, help="labeled JSONL (id, labeled_text)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    true_positive = predicted_positive = actual_positive = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits.argmax(dim=-1)
            mask = batch["labels"] != IGNORE_INDEX
            labels = batch["labels"][mask]
            predictions = predictions[mask]
            correct += int((predictions == labels).sum())
            total += int(mask.sum())
            true_positive += int(((predictions == 1) & (labels == 1)).sum())
            predicted_positive += int((predictions == 1).sum())
            actual_positive += int((labels == 1).sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": correct / max(1, total), "precision": precision, "recall": recall, "f1": f1}


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    examples = load_labeled_jsonl(args.train_data)
    random.shuffle(examples)
    n_validation = max(1, int(len(examples) * args.validation_fraction))
    validation_examples, train_examples = examples[:n_validation], examples[n_validation:]

    train_dataset = WindowedTokenDataset(train_examples, tokenizer, args.max_length, args.stride)
    validation_dataset = WindowedTokenDataset(validation_examples, tokenizer, args.max_length, args.stride)
    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "DATA", 1: "INSTRUCTION"},
        label2id={"DATA": 0, "INSTRUCTION": 1},
    ).to(device)

    class_weights = train_dataset.class_weights().to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    output_dir = Path(args.output_dir)
    best_dir = output_dir / "best"
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits.view(-1, 2), batch["labels"].view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}", flush=True)
        metrics = evaluate(model, validation_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(1, len(train_loader))
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"best validation f1 {best_f1:.4f}; checkpoint at {best_dir}")


if __name__ == "__main__":
    main()
