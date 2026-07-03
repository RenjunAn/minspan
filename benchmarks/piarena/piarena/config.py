import yaml
import json
import os


def load_experiment_config(config_path: str | None) -> dict:
    """Load a YAML experiment config file."""
    if config_path is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")
    return config


def merge_config_args(args, file_config: dict, defaults: dict) -> None:
    """Apply YAML/default values to argparse args in-place."""
    for key, default_val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, file_config.get(key, default_val))
    args.attack_config = file_config.get("attack_config", None)
    args.defense_config = file_config.get("defense_config", None)


def load_json_env_config(env_var: str) -> dict | None:
    """Load a JSON object from an environment variable."""
    raw_config = os.environ.get(env_var)
    if not raw_config:
        return None
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_var} must be valid JSON.") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{env_var} must decode to a JSON object.")
    return config
