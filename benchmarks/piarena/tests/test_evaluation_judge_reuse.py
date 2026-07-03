from piarena.evaluations import (
    DEFAULT_JUDGE_MODEL,
    get_reusable_judge_llm_kwargs,
    llm_judge,
    substring_match,
)


def test_reuses_loaded_backend_for_default_llm_judge():
    loaded_llm = object()

    assert get_reusable_judge_llm_kwargs(llm_judge, DEFAULT_JUDGE_MODEL, loaded_llm) == {
        "llm": loaded_llm,
    }


def test_does_not_reuse_backend_for_non_judge_or_non_default_model():
    loaded_llm = object()

    assert get_reusable_judge_llm_kwargs(substring_match, DEFAULT_JUDGE_MODEL, loaded_llm) == {}
    assert get_reusable_judge_llm_kwargs(llm_judge, "gpt-4o", loaded_llm) == {}


def test_llm_judge_uses_short_deterministic_generation():
    class RecordingLLM:
        def __init__(self):
            self.calls = []

        def query(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "YES"

    loaded_llm = RecordingLLM()

    assert llm_judge("done", ground_truth="unused", task_prompt="task", llm=loaded_llm) is True
    assert loaded_llm.calls[0][1] == {"max_new_tokens": 10, "temperature": 0.0}


def test_prefers_explicit_judge_backend_over_loaded_backend():
    loaded_llm = object()
    strategy_judge_llm = object()

    assert get_reusable_judge_llm_kwargs(
        llm_judge,
        DEFAULT_JUDGE_MODEL,
        loaded_llm,
        preferred_llm=strategy_judge_llm,
    ) == {"llm": strategy_judge_llm}
