"""YAML config loading with `section.key=value` command-line overrides."""

import yaml


def apply(cfg, overrides):
    """Apply overrides like "optim.lr=3e-4", cast to the type of the existing value.
    Only bool / str / int / float values are handled; other types are left unchanged."""
    for over in overrides:
        updates = over.partition("=")
        key1, key2 = updates[0].split(".")
        old_val = cfg[key1][key2]
        new_val_str = updates[-1]
        # check bool before int: isinstance(True, int) is True
        if isinstance(old_val, bool):
            cfg[key1][key2] = new_val_str.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(old_val, str):
            cfg[key1][key2] = new_val_str
        elif isinstance(old_val, int):
            cfg[key1][key2] = int(new_val_str)
        elif isinstance(old_val, float):
            cfg[key1][key2] = float(new_val_str)
    return cfg


def load_config(repo_root, path, overrides=[]):
    """Load a YAML config, apply overrides, and resolve `paths.*` and `logging.out_dir`
    relative to the repo root so scripts work from any working directory."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides != []:
        cfg = apply(cfg, overrides)
    if "logging" in cfg:
        # training configs only
        cfg["logging"]["out_dir"] = repo_root / cfg["logging"]["out_dir"]
    if "paths" in cfg:
        for k, v in cfg["paths"].items():
            cfg["paths"][k] = repo_root / v
    return cfg
