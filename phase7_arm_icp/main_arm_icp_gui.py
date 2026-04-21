"""Phase7 GUI entrypoint: live cloud + virtual arm overlay + cpi solve button."""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase7_arm_icp.src.tk_gui import run_phase7_arm_icp_gui


def load_config(config_path: str) -> dict:
    path = config_path
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase7 arm ICP GUI")
    p.add_argument("--config", default="config/settings.yaml", help="settings.yaml path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    settings_path = args.config
    if not os.path.isabs(settings_path):
        settings_path = os.path.join(_ROOT, settings_path)

    run_phase7_arm_icp_gui(cfg=cfg, root_dir=_ROOT, settings_path=settings_path)


if __name__ == "__main__":
    main()
