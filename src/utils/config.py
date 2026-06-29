"""配置加载工具:读取 config.yaml,解析路径,提供安全的嵌套取值。

所有步骤(s1..s8)共用本模块加载配置,避免各处重复解析。
"""
from __future__ import annotations

import os
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "缺少依赖 pyyaml,请先安装:pip install pyyaml"
    ) from e


def project_root() -> str:
    """返回项目根目录(本文件位于 <root>/src/utils/config.py)。"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config(path: str | None = None) -> dict[str, Any]:
    """加载 YAML 配置。

    path 为 None 时默认读取项目根目录的 config.yaml。
    返回的 dict 额外注入 ``_root``(项目根)与 ``_config_path`` 便于路径解析。
    """
    if path is None:
        path = os.path.join(project_root(), "config.yaml")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"找不到配置文件:{path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_root"] = project_root()
    cfg["_config_path"] = path
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """按点号路径取嵌套值,如 get(cfg, 'beats.subdivide', 1)。缺失返回 default。"""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path(cfg: dict, rel: str) -> str:
    """把配置里的相对路径解析为相对项目根的绝对路径。"""
    if os.path.isabs(rel):
        return rel
    return os.path.abspath(os.path.join(cfg.get("_root", project_root()), rel))
