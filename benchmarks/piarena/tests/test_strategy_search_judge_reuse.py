from piarena.attacks.strategy_search import attack_strategy_search
from piarena.evaluations import DEFAULT_JUDGE_MODEL, llm_judge, substring_match


class FakeLLM:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path


def test_cached_strategy_search_judge_prefers_matching_judge_backend():
    original_judge = attack_strategy_search.JUDGE_VLLM_MODEL
    original_attacker = attack_strategy_search.ATTACKER_VLLM_MODEL
    judge_llm = FakeLLM(DEFAULT_JUDGE_MODEL)
    attacker_llm = FakeLLM(DEFAULT_JUDGE_MODEL)
    try:
        attack_strategy_search.JUDGE_VLLM_MODEL = judge_llm
        attack_strategy_search.ATTACKER_VLLM_MODEL = attacker_llm

        assert attack_strategy_search.get_cached_strategy_search_judge_llm() is judge_llm
    finally:
        attack_strategy_search.JUDGE_VLLM_MODEL = original_judge
        attack_strategy_search.ATTACKER_VLLM_MODEL = original_attacker


def test_cached_strategy_search_judge_falls_back_to_matching_attacker_backend():
    original_judge = attack_strategy_search.JUDGE_VLLM_MODEL
    original_attacker = attack_strategy_search.ATTACKER_VLLM_MODEL
    attacker_llm = FakeLLM(DEFAULT_JUDGE_MODEL)
    try:
        attack_strategy_search.JUDGE_VLLM_MODEL = None
        attack_strategy_search.ATTACKER_VLLM_MODEL = attacker_llm

        assert attack_strategy_search.get_cached_strategy_search_judge_llm() is attacker_llm
    finally:
        attack_strategy_search.JUDGE_VLLM_MODEL = original_judge
        attack_strategy_search.ATTACKER_VLLM_MODEL = original_attacker


def test_judge_fallback_reuses_default_target_before_lazy_attacker_load():
    attacker = object.__new__(attack_strategy_search.StrategySearchAttacker)
    target_llm = FakeLLM(DEFAULT_JUDGE_MODEL)
    attacker.target_llm = target_llm
    attacker.attacker_llm = None
    attacker.attacker_model_name_or_path = DEFAULT_JUDGE_MODEL
    attacker._ensure_attacker_llm = lambda: (_ for _ in ()).throw(
        AssertionError("should not lazy-load attacker judge")
    )

    assert attacker._get_judge_fallback_llm() is target_llm


def test_llm_judge_kwargs_prefers_strategy_search_vllm():
    attacker = object.__new__(attack_strategy_search.StrategySearchAttacker)
    judge_llm = FakeLLM(DEFAULT_JUDGE_MODEL)
    attacker.judge_vllm = judge_llm

    assert attacker._get_llm_judge_kwargs(llm_judge) == {"llm": judge_llm}
    assert attacker._get_llm_judge_kwargs(substring_match) == {}
