"""Training CLI for frozen DataFilter token-classification heads."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from minspan.metrics import MetricsAccumulator
from minspan.training_data import TaggerCollator
from minspan.modeling import (
    EncoderTagger,
    load_datafilter_backbone,
    load_encoder_tagger,
    load_tagger_head,
    save_encoder_checkpoint,
    save_tagger_checkpoint,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MinSpan token tagger")
    parser.add_argument("--train-data", default="data/train.jsonl")
    parser.add_argument("--validation-data", default="data/validation.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default="JoyYizhu/DataFilter")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument(
        "--backbone-type",
        choices=["datafilter", "encoder"],
        default="datafilter",
        help=(
            "datafilter: frozen DataFilter backbone with a trainable head; "
            "encoder: fully fine-tuned token-classification encoder"
        ),
    )
    parser.add_argument(
        "--head-type",
        choices=["linear", "bidir_transformer"],
        default="linear",
    )
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--num-transformer-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=1.0,
        help="Loss weight for tokens near KEEP/DROP transitions (encoder only)",
    )
    parser.add_argument(
        "--boundary-radius",
        type=int,
        default=2,
        help="Token radius around a transition that receives boundary weight",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--instruction-dropout",
        type=float,
        default=0.0,
        help="Probability of blanking the instruction per training record "
        "(applied to the training collator only; Phase 2 default: 0.15)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        if args.backbone_type == "encoder":
            model_basename = args.model_name.split("/")[-1].lower()
            args.output_dir = f"outputs/tagger-encoder-{model_basename}"
        elif args.head_type == "linear":
            args.output_dir = "outputs/tagger-linear"
        else:
            args.output_dir = (
                f"outputs/tagger-bidir-{args.num_transformer_layers}l-"
                f"{args.projection_dim}"
            )
    if args.learning_rate is None:
        if args.backbone_type == "encoder":
            args.learning_rate = 3e-5
        else:
            args.learning_rate = 1e-3 if args.head_type == "linear" else 1e-4
    return args


def _trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
    if hasattr(model, "parameters"):
        candidates = model.parameters()
    elif hasattr(model, "classifier"):
        candidates = model.classifier.parameters()
    else:
        candidates = ()
    parameters = [
        parameter for parameter in candidates if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def build_optimizer(
    model: Any,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        _trainable_parameters(model),
        lr=lr,
        weight_decay=weight_decay,
    )


def _batch_context(batch: dict[str, Any]) -> str:
    return (
        f"record_ids={batch.get('record_ids', [])}, "
        f"token_lengths={batch.get('token_lengths', [])}"
    )


def _rescale_partial_gradients(
    parameters,
    gradient_accumulation_steps: int,
    pending_batches: int,
) -> None:
    if pending_batches == gradient_accumulation_steps:
        return
    scale = gradient_accumulation_steps / pending_batches
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def train_one_epoch(
    model: Any,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    gradient_accumulation_steps: int,
    log_interval: int = 10,
) -> dict[str, Any]:
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if log_interval < 1:
        raise ValueError("log_interval must be >= 1")

    model.train()
    trainable_parameters = _trainable_parameters(model)
    optimizer.zero_grad()
    total_loss = 0.0
    num_batches = 0
    optimizer_steps = 0
    processed_examples = 0
    processed_tokens = 0
    pending_batches = 0
    started_at = time.time()
    total_batches = len(dataloader)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        try:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                f"CUDA OOM while training {_batch_context(batch)}"
            ) from error

        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(
                f"NaN/Inf loss on batch with {_batch_context(batch)}"
            )

        (loss / gradient_accumulation_steps).backward()
        pending_batches += 1
        total_loss += loss.item()
        num_batches += 1
        processed_examples += input_ids.shape[0]
        processed_tokens += int(attention_mask.sum().item())

        is_last_batch = batch_index == total_batches - 1
        accumulation_boundary = (
            pending_batches == gradient_accumulation_steps
        )
        if accumulation_boundary or is_last_batch:
            _rescale_partial_gradients(
                trainable_parameters,
                gradient_accumulation_steps,
                pending_batches,
            )
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_steps += 1
            pending_batches = 0

            if optimizer_steps % log_interval == 0:
                current_lr = (
                    scheduler.get_last_lr()[0]
                    if hasattr(scheduler, "get_last_lr")
                    else 0.0
                )
                print(
                    f"step={optimizer_steps} "
                    f"loss={total_loss / num_batches:.4f} "
                    f"lr={current_lr:.2e} "
                    f"examples={processed_examples} "
                    f"tokens={processed_tokens} "
                    f"elapsed={time.time() - started_at:.1f}s"
                )

    if num_batches == 0:
        raise ValueError("training dataloader is empty")

    peak_vram_bytes = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    return {
        "optimizer_steps": optimizer_steps,
        "loss": total_loss / num_batches,
        "processed_examples": processed_examples,
        "processed_tokens": processed_tokens,
        "elapsed_seconds": time.time() - started_at,
        "peak_vram_bytes": peak_vram_bytes,
    }


def validate_epoch(
    model: Any,
    dataloader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    accumulator = MetricsAccumulator()
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            try:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    f"CUDA OOM while validating {_batch_context(batch)}"
                ) from error

            predictions = output.logits.argmax(dim=-1).cpu()
            accumulator.add_batch(
                predictions,
                batch["labels"],
                batch["records"],
                token_lengths=batch.get("token_lengths"),
                offset_mapping=batch["offset_mapping"],
                data_ranges=batch["data_ranges"],
            )
            batches += 1

    if batches == 0:
        raise ValueError("validation dataloader is empty")
    return accumulator.compute()


def _training_metadata(args: argparse.Namespace, epoch: int, **extra) -> dict[str, Any]:
    return {
        "base_model_name": getattr(args, "model_name", "unknown"),
        "base_model_revision": getattr(args, "model_revision", None),
        "epoch": epoch,
        "training_arguments": vars(args).copy(),
        **extra,
    }


def _save_checkpoint(model: Any, tokenizer: Any, directory: Path, metadata) -> None:
    if isinstance(model, EncoderTagger):
        save_encoder_checkpoint(model, tokenizer, directory, metadata)
    else:
        save_tagger_checkpoint(model, tokenizer, directory, metadata)


def _verify_checkpoint_forward(
    checkpoint_dir: Path,
    model: Any,
    validation_loader,
    device: torch.device,
) -> None:
    if isinstance(model, EncoderTagger):
        reloaded = load_encoder_tagger(checkpoint_dir).to(device)
    else:
        reloaded = load_tagger_head(model.backbone, checkpoint_dir).to(device)
    reloaded.eval()
    batch = next(iter(validation_loader))
    with torch.no_grad():
        logits = reloaded(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
    if not torch.isfinite(logits).all():
        raise FloatingPointError("reloaded checkpoint produced NaN/Inf logits")


def build_collators(
    tokenizer: Any,
    prompt_builder: Any,
    args: argparse.Namespace,
) -> tuple[TaggerCollator, TaggerCollator]:
    """Training collator (with instruction dropout) and a clean validation
    collator."""
    train_collator = TaggerCollator(
        tokenizer,
        prompt_builder=prompt_builder,
        instruction_dropout=args.instruction_dropout,
        dropout_seed=args.seed,
    )
    validation_collator = TaggerCollator(tokenizer, prompt_builder=prompt_builder)
    return train_collator, validation_collator


def run_training(
    model: Any,
    tokenizer: Any,
    train_dataset,
    val_dataset,
    collate_fn,
    output_dir: Path,
    args: argparse.Namespace,
    val_collate_fn=None,
) -> dict[str, Any]:
    if args.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {args.batch_size}")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collate_fn if val_collate_fn is not None else collate_fn,
        pin_memory=pin_memory,
    )
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("training and validation datasets must be non-empty")

    optimizer = build_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (
            (current_step - warmup_steps)
            / max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_drop_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_interval=args.log_interval,
        )
        validation_metrics = validate_epoch(
            model,
            validation_loader,
            device=device,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train": train_result,
            "val": validation_metrics,
        }
        (output_dir / f"metrics_{epoch}.json").write_text(
            json.dumps(epoch_metrics, indent=2),
            encoding="utf-8",
        )

        drop_f1 = validation_metrics["overall"]["drop"]["f1"]
        if drop_f1 >= best_drop_f1:
            best_drop_f1 = drop_f1
            best_epoch = epoch
            _save_checkpoint(
                model,
                tokenizer,
                output_dir / "best",
                _training_metadata(
                    args,
                    epoch,
                    drop_f1=drop_f1,
                ),
            )

    _save_checkpoint(
        model,
        tokenizer,
        output_dir / "last",
        _training_metadata(args, args.epochs),
    )
    _verify_checkpoint_forward(
        output_dir / "best",
        model,
        validation_loader,
        device,
    )
    return {
        "best_epoch": best_epoch,
        "best_drop_f1": best_drop_f1,
    }


def main() -> None:
    from transformers import AutoConfig, AutoTokenizer

    from minspan.training_data import JsonlTaggerDataset, inspect_dataset
    from minspan.modeling import FrozenDataFilterTagger, load_encoder_backbone
    from minspan.prompting import build_encoder_prompt, build_tagger_prompt

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("token alignment requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer must define a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = JsonlTaggerDataset(
        args.train_data,
        limit=args.max_train_samples,
    )
    validation_dataset = JsonlTaggerDataset(
        args.validation_data,
        limit=args.max_validation_samples,
    )
    config = AutoConfig.from_pretrained(
        args.model_name,
        revision=args.model_revision,
    )
    model_max_length = int(
        getattr(config, "max_position_embeddings", tokenizer.model_max_length)
    )
    prompt_builder = (
        build_encoder_prompt
        if args.backbone_type == "encoder"
        else build_tagger_prompt
    )
    for split, dataset in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):
        stats = inspect_dataset(
            dataset,
            tokenizer,
            model_max_length=model_max_length,
            prompt_builder=prompt_builder,
        )
        print(f"{split} token lengths: {json.dumps(stats)}")

    if args.backbone_type == "encoder":
        model = load_encoder_backbone(
            args.model_name,
            revision=args.model_revision,
            boundary_weight=args.boundary_weight,
            boundary_radius=args.boundary_radius,
        )
    else:
        backbone = load_datafilter_backbone(
            args.model_name,
            revision=args.model_revision,
        )
        model = FrozenDataFilterTagger(
            backbone,
            head_type=args.head_type,
            projection_dim=args.projection_dim,
            num_attention_heads=args.num_attention_heads,
            ffn_dim=args.ffn_dim,
            num_transformer_layers=args.num_transformer_layers,
            dropout=args.dropout,
        )
    print(json.dumps(model.architecture_config(), indent=2))
    train_collator, validation_collator = build_collators(
        tokenizer, prompt_builder, args
    )
    result = run_training(
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
        train_collator,
        Path(args.output_dir),
        args,
        val_collate_fn=validation_collator,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
