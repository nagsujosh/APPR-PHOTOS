import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    # Support _base_ key for inheritance
    if "_base_" in cfg:
        base_path = path.parent / cfg.pop("_base_")
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides like 'training.lr=0.001'."""
    for override in overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to parse value as yaml scalar
        try:
            d[keys[-1]] = yaml.safe_load(value)
        except yaml.YAMLError:
            d[keys[-1]] = value
    return cfg


def get_config(args: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args(args)

    cfg = load_config(parsed.config)
    if parsed.overrides:
        cfg = apply_overrides(cfg, parsed.overrides)
    return cfg


class Config:
    """Dot-access wrapper around a dict."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __repr__(self):
        return repr(self.__dict__)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
