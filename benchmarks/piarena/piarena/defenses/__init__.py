from .base import BaseDefense
from ..registry import DEFENSE_REGISTRY
import importlib


_DEFENSE_MODULES = {
    "none": ".defense_none",
    "secalign": ".secalign.defense_secalign",
    "pisanitizer": ".pisanitizer.defense_pisanitizer",
    "datasentinel": ".datasentinel.defense_datasentinel",
    "attentiontracker": ".attentiontracker.defense_attentiontracker",
    "promptguard": ".promptguard.defense_promptguard",
    "promptlocate": ".promptlocate.defense_promptlocate",
    "promptarmor": ".promptarmor.defense_promptarmor",
    "datafilter": ".datafilter.defense_datafilter",
    "piguard": ".piguard.defense_piguard",
    "deepseek_pisanitizer": ".deepseek_pisanitizer.defense_deepseek_pisanitizer",
    "modernbert_tagger": ".modernbert_tagger.defense_modernbert_tagger",
    "commandsans": ".commandsans.defense_commandsans",
}

_IMPORT_ERRORS = {}


def _import_defense(name: str, *, raise_error: bool = False) -> None:
    module_path = _DEFENSE_MODULES[name]
    try:
        importlib.import_module(module_path, __name__)
    except ImportError as exc:
        _IMPORT_ERRORS[name] = exc
        if raise_error:
            raise ImportError(f"Could not import defense {name!r}: {exc}") from exc


# Import defense modules to trigger @register_defense decorators when dependencies
# are available. Missing optional dependencies are reported when that defense is
# requested through get_defense().
for _defense_name in _DEFENSE_MODULES:
    _import_defense(_defense_name)


# Registry is auto-populated by @register_defense decorators
DEFENSE_CLASSES = DEFENSE_REGISTRY


def get_defense(name: str, config: dict = None) -> BaseDefense:
    """Factory: instantiate a defense by name."""
    try:
        cls = DEFENSE_REGISTRY.get(name)
    except ValueError:
        if name in _DEFENSE_MODULES:
            _import_defense(name, raise_error=True)
            cls = DEFENSE_REGISTRY.get(name)
        else:
            raise ValueError(f"Unknown defense: {name}. Available: {sorted(DEFENSE_REGISTRY)}")
    return cls(config=config)
