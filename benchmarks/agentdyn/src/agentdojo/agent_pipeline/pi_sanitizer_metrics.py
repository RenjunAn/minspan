from __future__ import annotations

from difflib import SequenceMatcher


def removed_fragments(original: str, filtered: str) -> list[str]:
    fragments = []
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(None, original, filtered).get_opcodes():
        if tag in {"delete", "replace"} and i1 != i2:
            fragments.append(original[i1:i2])
    return fragments


def normalized_edit_similarity(left: str, right: str, edit_distance: int | None = None) -> float:
    if edit_distance is None:
        edit_distance = levenshtein_distance(left, right)
    denominator = max(len(left), len(right), 1)
    return _clamp_score(1.0 - (edit_distance / denominator))


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))
