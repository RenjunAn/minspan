import json
from argparse import Namespace
from pathlib import Path

from agentdojo.agent_pipeline.pi_sanitizer_direct_api import build_run_config, write_run_config


def test_write_run_config_records_compare_settings_without_api_key(tmp_path):
    args = Namespace(
        runner="direct",
        data=Path("data/pi_detector/val.jsonl"),
        optimized_program=Path("runs/pi_sanitizer_gepa_smoke_60_sas/optimized_program.json"),
        output_dir=tmp_path,
        model="openai/deepseek-v4-flash",
        api_base="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        thinking="disabled",
        temperature=0.0,
        limit=3,
    )

    config = build_run_config(
        args,
        num_examples=3,
        extra_body={"thinking": {"type": "disabled"}},
        seed_prompt="seed prompt",
        optimized_prompt="optimized prompt",
    )
    write_run_config(tmp_path, config)

    saved = json.loads((tmp_path / "run_config.json").read_text())
    assert saved["runner"] == "direct"
    assert saved["model"] == "openai/deepseek-v4-flash"
    assert saved["direct_api_model"] == "deepseek-v4-flash"
    assert saved["extra_body"] == {"thinking": {"type": "disabled"}}
    assert saved["num_examples"] == 3
    assert saved["seed_prompt"] == "seed prompt"
    assert saved["optimized_prompt"] == "optimized prompt"
    assert len(saved["seed_prompt_sha256"]) == 64
    assert len(saved["optimized_prompt_sha256"]) == 64
    assert "api_key" not in saved
