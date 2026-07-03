"""Token, character, reconstruction, and grouped metrics for the tagger."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

KEEP_LABEL = 0
DROP_LABEL = 1
IGNORE_LABEL = -100


def _to_int_list(seq: Any) -> list[int]:
    try:
        values = seq.detach().cpu().reshape(-1).tolist()
    except AttributeError:
        try:
            values = seq.reshape(-1).tolist()
        except AttributeError:
            values = list(seq)
    if values and isinstance(values[0], (list, tuple)):
        return [value for row in values for value in row]
    return [int(value) for value in values]


def _to_rows(seq: Any) -> list[list[int]]:
    try:
        values = seq.detach().cpu().tolist()
    except AttributeError:
        try:
            values = seq.tolist()
        except AttributeError:
            values = list(seq)
    if not values:
        return []
    if isinstance(values[0], (list, tuple)):
        return [[int(value) for value in row] for row in values]
    return [[int(value) for value in values]]


def _to_offset_rows(seq: Any) -> list[list[tuple[int, int]]]:
    try:
        values = seq.detach().cpu().tolist()
    except AttributeError:
        try:
            values = seq.tolist()
        except AttributeError:
            values = list(seq)
    if not values:
        return []
    if values and values[0] and isinstance(values[0][0], int):
        values = [values]
    return [
        [(int(start), int(end)) for start, end in row]
        for row in values
    ]


def _class_metrics(tp: int, fp: int, fn: int, support: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


class _TokenCounts:
    def __init__(self):
        self.keep_tp = 0
        self.keep_fp = 0
        self.keep_fn = 0
        self.keep_support = 0
        self.drop_tp = 0
        self.drop_fp = 0
        self.drop_fn = 0
        self.drop_support = 0
        self.correct = 0
        self.total = 0

    def add(self, prediction: int, label: int) -> None:
        if label == IGNORE_LABEL:
            return
        if label not in (KEEP_LABEL, DROP_LABEL):
            raise ValueError(f"unexpected token label: {label}")
        if prediction not in (KEEP_LABEL, DROP_LABEL):
            raise ValueError(f"unexpected token prediction: {prediction}")

        self.total += 1
        self.correct += int(prediction == label)
        if label == KEEP_LABEL:
            self.keep_support += 1
            if prediction == KEEP_LABEL:
                self.keep_tp += 1
            else:
                self.keep_fn += 1
                self.drop_fp += 1
        else:
            self.drop_support += 1
            if prediction == DROP_LABEL:
                self.drop_tp += 1
            else:
                self.drop_fn += 1
                self.keep_fp += 1

    def metrics(self) -> dict[str, Any]:
        keep = _class_metrics(
            self.keep_tp,
            self.keep_fp,
            self.keep_fn,
            self.keep_support,
        )
        drop = _class_metrics(
            self.drop_tp,
            self.drop_fp,
            self.drop_fn,
            self.drop_support,
        )
        return {
            "keep": keep,
            "drop": drop,
            "macro_f1": (keep["f1"] + drop["f1"]) / 2.0,
            "accuracy": self.correct / self.total if self.total else 0.0,
            "confusion_matrix": {
                "true_keep_pred_keep": self.keep_tp,
                "true_keep_pred_drop": self.keep_fn,
                "true_drop_pred_keep": self.drop_fn,
                "true_drop_pred_drop": self.drop_tp,
            },
        }


def compute_token_metrics(predictions: Any, labels: Any) -> dict[str, Any]:
    preds = _to_int_list(predictions)
    labs = _to_int_list(labels)
    if len(preds) != len(labs):
        raise ValueError(
            f"prediction count {len(preds)} != label count {len(labs)}"
        )
    counts = _TokenCounts()
    for prediction, label in zip(preds, labs):
        counts.add(prediction, label)
    return counts.metrics()


def prediction_to_data_spans(
    predictions: Any,
    offset_mapping: Sequence[tuple[int, int]],
    data_start: int,
    data_end: int,
) -> list[dict[str, int]]:
    preds = _to_int_list(predictions)
    spans = []
    for prediction, (token_start, token_end) in zip(preds, offset_mapping):
        if prediction != DROP_LABEL or token_start == token_end:
            continue
        overlap_start = max(token_start, data_start)
        overlap_end = min(token_end, data_end)
        if overlap_start < overlap_end:
            spans.append(
                {
                    "start": overlap_start - data_start,
                    "end": overlap_end - data_start,
                }
            )
    return spans


def _merge_spans(spans: Sequence[dict[str, int]]) -> list[dict[str, int]]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda span: (span["start"], span["end"]))
    merged = [dict(sorted_spans[0])]
    for span in sorted_spans[1:]:
        previous = merged[-1]
        if span["start"] <= previous["end"]:
            previous["end"] = max(previous["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


def reconstruct_clean_text(
    attacked_data: str,
    drop_spans: Sequence[dict[str, int]],
) -> str:
    result = []
    cursor = 0
    for span in _merge_spans(drop_spans):
        result.append(attacked_data[cursor:span["start"]])
        cursor = span["end"]
    result.append(attacked_data[cursor:])
    return "".join(result)


def _span_length(spans: Sequence[dict[str, int]]) -> int:
    return sum(span["end"] - span["start"] for span in _merge_spans(spans))


def _intersection_length(
    left: Sequence[dict[str, int]],
    right: Sequence[dict[str, int]],
) -> int:
    left_merged = _merge_spans(left)
    right_merged = _merge_spans(right)
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_span = left_merged[left_index]
        right_span = right_merged[right_index]
        total += max(
            0,
            min(left_span["end"], right_span["end"])
            - max(left_span["start"], right_span["start"]),
        )
        if left_span["end"] <= right_span["end"]:
            left_index += 1
        else:
            right_index += 1
    return total


class _MetricGroup:
    def __init__(self):
        self.tokens = _TokenCounts()
        self.character_tp = 0
        self.character_fp = 0
        self.character_fn = 0
        self.exact_matches = 0
        self.records = 0

    def add_tokens(self, predictions: Sequence[int], labels: Sequence[int]) -> None:
        for prediction, label in zip(predictions, labels):
            self.tokens.add(int(prediction), int(label))

    def add_record(
        self,
        predictions: Sequence[int],
        labels: Sequence[int],
        record: dict[str, Any],
        offsets: Sequence[tuple[int, int]],
        data_range: tuple[int, int],
    ) -> None:
        self.add_tokens(predictions, labels)
        predicted_spans = prediction_to_data_spans(
            predictions,
            offsets,
            data_range[0],
            data_range[1],
        )
        true_spans = record["drop_spans"]
        true_positive = _intersection_length(predicted_spans, true_spans)
        self.character_tp += true_positive
        self.character_fp += _span_length(predicted_spans) - true_positive
        self.character_fn += _span_length(true_spans) - true_positive
        reconstructed = reconstruct_clean_text(
            record["attacked_data"],
            predicted_spans,
        )
        self.exact_matches += int(reconstructed == record["clean_data"])
        self.records += 1

    def metrics(self) -> dict[str, Any]:
        result = self.tokens.metrics()
        character_support = self.character_tp + self.character_fn
        result["character_drop"] = _class_metrics(
            self.character_tp,
            self.character_fp,
            self.character_fn,
            character_support,
        )
        result["exact_clean_match"] = (
            self.exact_matches / self.records if self.records else 0.0
        )
        result["records"] = self.records
        return result


class MetricsAccumulator:
    def __init__(self):
        self._overall = _MetricGroup()
        self._by_attack_type: dict[str, _MetricGroup] = defaultdict(_MetricGroup)
        self._by_position: dict[str, _MetricGroup] = defaultdict(_MetricGroup)
        self._by_cut_type: dict[str, _MetricGroup] = defaultdict(_MetricGroup)

    def _groups_for_record(self, record: dict[str, Any]) -> list[_MetricGroup]:
        groups = [self._overall]
        for field, mapping in (
            ("attack_type", self._by_attack_type),
            ("position", self._by_position),
            ("cut_type", self._by_cut_type),
        ):
            value = record.get(field)
            if value is not None:
                groups.append(mapping[str(value)])
        return groups

    def add_batch(
        self,
        predictions: Any,
        labels: Any,
        records: list[dict[str, Any]],
        token_lengths: list[int] | None = None,
        offset_mapping: Any | None = None,
        data_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        if offset_mapping is not None or data_ranges is not None:
            if offset_mapping is None or data_ranges is None:
                raise ValueError("offset_mapping and data_ranges must be provided together")
            prediction_rows = _to_rows(predictions)
            label_rows = _to_rows(labels)
            offset_rows = _to_offset_rows(offset_mapping)
            if not (
                len(prediction_rows)
                == len(label_rows)
                == len(offset_rows)
                == len(data_ranges)
                == len(records)
            ):
                raise ValueError("batch metric fields have inconsistent record counts")
            for prediction_row, label_row, offsets, data_range, record in zip(
                prediction_rows,
                label_rows,
                offset_rows,
                data_ranges,
                records,
            ):
                if not (
                    len(prediction_row) == len(label_row) == len(offsets)
                ):
                    raise ValueError("batch metric fields have inconsistent token counts")
                for group in self._groups_for_record(record):
                    group.add_record(
                        prediction_row,
                        label_row,
                        record,
                        offsets,
                        data_range,
                    )
            return

        preds = _to_int_list(predictions)
        labs = _to_int_list(labels)
        if len(preds) != len(labs):
            raise ValueError("prediction and label counts differ")
        if not records:
            self._overall.add_tokens(preds, labs)
            return
        for index, (prediction, label) in enumerate(zip(preds, labs)):
            record = records[index % len(records)]
            for group in self._groups_for_record(record):
                group.add_tokens([prediction], [label])

    def compute(self) -> dict[str, Any]:
        return {
            "overall": self._overall.metrics(),
            "by_attack_type": {
                key: value.metrics()
                for key, value in sorted(self._by_attack_type.items())
            },
            "by_position": {
                key: value.metrics()
                for key, value in sorted(self._by_position.items())
            },
            "by_cut_type": {
                key: value.metrics()
                for key, value in sorted(self._by_cut_type.items())
            },
        }
