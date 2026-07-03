import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import save_file, load_file

from minspan.prompting import ENCODER_PROMPT_FORMAT_VERSION, PROMPT_FORMAT_VERSION


class LinearTokenHead(nn.Linear):
    def forward(self, hidden, attention_mask=None):
        return super().forward(hidden)


class BidirectionalTransformerHead(nn.Module):
    def __init__(
        self,
        input_size,
        projection_dim=512,
        num_attention_heads=8,
        ffn_dim=2048,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        if projection_dim < 1:
            raise ValueError("projection_dim must be >= 1")
        if num_attention_heads < 1:
            raise ValueError("num_attention_heads must be >= 1")
        if projection_dim % num_attention_heads != 0:
            raise ValueError(
                "projection_dim must be divisible by num_attention_heads"
            )
        if ffn_dim < 1:
            raise ValueError("ffn_dim must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_projection = nn.Linear(input_size, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(projection_dim, 2)

    def forward(self, hidden, attention_mask=None):
        projected = self.input_norm(self.input_projection(hidden))
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        encoded = self.encoder(
            projected,
            src_key_padding_mask=padding_mask,
        )
        return self.classifier(encoded)


class FrozenDataFilterTagger(nn.Module):
    def __init__(
        self,
        backbone,
        *,
        head_type="linear",
        projection_dim=512,
        num_attention_heads=8,
        ffn_dim=2048,
        num_transformer_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.backbone.config.use_cache = False
        hidden_size = self.backbone.config.hidden_size
        first_parameter = next(self.backbone.parameters(), None)
        head_device = (
            first_parameter.device if first_parameter is not None else torch.device("cpu")
        )
        self.head_type = head_type
        self._head_config = {}

        if head_type == "linear":
            self.head = LinearTokenHead(hidden_size, 2)
        elif head_type == "bidir_transformer":
            self._head_config = {
                "projection_dim": projection_dim,
                "num_attention_heads": num_attention_heads,
                "ffn_dim": ffn_dim,
                "num_transformer_layers": num_transformer_layers,
                "dropout": dropout,
            }
            self.head = BidirectionalTransformerHead(
                input_size=hidden_size,
                projection_dim=projection_dim,
                num_attention_heads=num_attention_heads,
                ffn_dim=ffn_dim,
                num_layers=num_transformer_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unsupported head_type: {head_type}")

        self.head.to(device=head_device, dtype=torch.float32)

    @property
    def classifier(self):
        """Preserve the public attribute used by existing linear checkpoints/tests."""
        if self.head_type == "linear":
            return self.head
        return self.head.classifier

    @property
    def head_config(self):
        return self._head_config.copy()

    def trainable_parameters(self):
        return self.head.parameters()

    def architecture_config(self):
        return {
            "head_type": self.head_type,
            "hidden_size": self.backbone.config.hidden_size,
            "head_config": self.head_config,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.head.parameters()
            ),
        }

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state
        if hidden.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.head(hidden, attention_mask=attention_mask)
        else:
            logits = self.head(hidden.float(), attention_mask=attention_mask)
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 2),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return SimpleNamespace(logits=logits, loss=loss)


def boundary_weight_mask(labels, radius, weight):
    """Per-token loss weights: `weight` within `radius` tokens of a KEEP/DROP
    transition, 1 elsewhere, 0 on ignored positions. The sequence edges count
    as virtual KEEP, so injections starting or ending the data section produce
    a boundary there too."""
    weights = torch.zeros(labels.shape, dtype=torch.float32, device=labels.device)
    for row in range(labels.shape[0]):
        valid_indices = (labels[row] != -100).nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            continue
        valid_labels = labels[row][valid_indices]
        padded = torch.cat(
            [
                torch.zeros(1, dtype=valid_labels.dtype, device=labels.device),
                valid_labels,
                torch.zeros(1, dtype=valid_labels.dtype, device=labels.device),
            ]
        )
        # boundaries[i] is the transition between compressed tokens i-1 and i
        boundaries = (padded[1:] != padded[:-1]).nonzero(as_tuple=True)[0]
        row_weights = torch.ones(
            valid_labels.shape, dtype=torch.float32, device=labels.device
        )
        count = valid_labels.shape[0]
        for boundary in boundaries.tolist():
            start = max(0, boundary - radius)
            end = min(count, boundary + radius)
            row_weights[start:end] = weight
        weights[row][valid_indices] = row_weights
    return weights


class EncoderTagger(nn.Module):
    """Fully fine-tuned bidirectional encoder for KEEP/DROP token tagging."""

    def __init__(self, model, boundary_weight=1.0, boundary_radius=2):
        super().__init__()
        if boundary_weight < 1.0:
            raise ValueError("boundary_weight must be >= 1.0")
        if boundary_radius < 0:
            raise ValueError("boundary_radius must be >= 0")
        self.model = model
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius

    def trainable_parameters(self):
        return self.model.parameters()

    def architecture_config(self):
        return {
            "head_type": "encoder",
            "base_model_type": self.model.config.model_type,
            "hidden_size": self.model.config.hidden_size,
            "boundary_weight": self.boundary_weight,
            "boundary_radius": self.boundary_radius,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
        }

    def forward(self, input_ids, attention_mask=None, labels=None):
        model_labels = labels if self.boundary_weight == 1.0 else None
        if input_ids.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=model_labels,
                )
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=model_labels,
            )
        logits = output.logits.float()
        loss = output.loss
        if labels is not None and self.boundary_weight != 1.0:
            per_token = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            )
            weights = boundary_weight_mask(
                labels,
                self.boundary_radius,
                self.boundary_weight,
            ).reshape(-1)
            loss = (per_token * weights).sum() / weights.sum().clamp(min=1.0)
        return SimpleNamespace(logits=logits, loss=loss)


def load_encoder_backbone(model_name, revision=None, boundary_weight=1.0, boundary_radius=2):
    """Initialize a token-classification encoder for full fine-tuning."""
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        revision=revision,
        num_labels=2,
        label2id={"KEEP": 0, "DROP": 1},
        id2label={0: "KEEP", 1: "DROP"},
    )
    return EncoderTagger(
        model,
        boundary_weight=boundary_weight,
        boundary_radius=boundary_radius,
    )


def save_encoder_checkpoint(model, tokenizer, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model.model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

    config = {
        **metadata,
        **model.architecture_config(),
        "label2id": {"KEEP": 0, "DROP": 1},
        "id2label": {"0": "KEEP", "1": "DROP"},
        "prompt_format_version": ENCODER_PROMPT_FORMAT_VERSION,
    }
    (directory / "tagger_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def load_encoder_tagger(directory):
    from transformers import AutoModelForTokenClassification

    directory = Path(directory)
    config = json.loads((directory / "tagger_config.json").read_text(encoding="utf-8"))
    head_type = config.get("head_type")
    if head_type != "encoder":
        raise ValueError(
            f"checkpoint head_type {head_type!r} is not an encoder checkpoint"
        )
    model = AutoModelForTokenClassification.from_pretrained(directory)
    return EncoderTagger(model)


def load_datafilter_backbone(model_name, revision=None):
    """Load the backbone without the causal language-model output head."""
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    backbone.config.use_cache = False
    backbone.eval()
    return backbone


def save_tagger_checkpoint(model, tokenizer, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.head.state_dict().items()
    }
    save_file(tensors, directory / "tagger_head.safetensors")

    config = {
        **metadata,
        "hidden_size": model.backbone.config.hidden_size,
        "head_type": model.head_type,
        "head_config": model.head_config,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.head.parameters()
        ),
        "label2id": {"KEEP": 0, "DROP": 1},
        "id2label": {"0": "KEEP", "1": "DROP"},
        "prompt_format_version": PROMPT_FORMAT_VERSION,
    }
    (directory / "tagger_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    tokenizer.save_pretrained(directory)


def load_tagger_head(backbone, directory):
    directory = Path(directory)
    config = json.loads((directory / "tagger_config.json").read_text(encoding="utf-8"))

    if backbone.config.hidden_size != config["hidden_size"]:
        raise ValueError(
            f"backbone hidden_size {backbone.config.hidden_size} != "
            f"checkpoint hidden_size {config['hidden_size']}"
        )

    head_type = config.get("head_type", "linear")
    head_config = config.get("head_config", {})
    model = FrozenDataFilterTagger(
        backbone,
        head_type=head_type,
        projection_dim=head_config.get("projection_dim", 512),
        num_attention_heads=head_config.get("num_attention_heads", 8),
        ffn_dim=head_config.get("ffn_dim", 2048),
        num_transformer_layers=head_config.get("num_transformer_layers", 1),
        dropout=head_config.get("dropout", 0.1),
    )

    tensors = load_file(directory / "tagger_head.safetensors")
    if head_type == "linear" and "classifier.weight" in tensors:
        tensors = {
            "weight": tensors["classifier.weight"],
            "bias": tensors["classifier.bias"],
        }
    model.head.load_state_dict(tensors, strict=True)

    return model
