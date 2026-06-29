"""标准件契约:统一的产物路径、step 规格与 manifest 读写。

让每个步骤(无论本地还是付费产品)都读写同一套标准件,实现可替换。
详见 docs/ARTIFACTS.md。
"""
from __future__ import annotations

import glob
import json
import os

SCHEMA = "1"

# 各步骤的标准输入/输出(逻辑名),用于 provider/handoff 通用处理。
# kind: frames(帧序列目录) | json | video
STEP_SPECS: dict[str, dict] = {
    "beats":    {"module": "s1_beats",           "inputs": [],                 "output": "beats",   "kind": "json"},
    "camera":   {"module": "s2_camera",          "inputs": ["beats"],          "output": "locked",  "kind": "frames"},
    "matte":    {"module": "s3_segment",         "inputs": ["locked"],         "output": "alpha",   "kind": "frames"},
    "plates":   {"module": "s4_plates",          "inputs": ["locked"],         "output": "plates",  "kind": "frames"},
    "garment":  {"module": "s5_garment",         "inputs": ["locked", "alpha"],"output": "garment", "kind": "frames"},
    "relight":  {"module": "s6_relight",         "inputs": ["garment"],        "output": "relit",   "kind": "frames"},
    "composite":{"module": "s7_composite",       "inputs": ["relit", "alpha", "plates"], "output": "comp", "kind": "frames"},
    "assemble": {"module": "s8_restore_assemble","inputs": ["comp", "camera"], "output": "final",   "kind": "video"},
}

# 逻辑名 → 相对 data/work 的目录/文件
_ARTIFACT_DIRS = {
    "src": "src", "locked": "locked", "alpha": "alpha",
    "plates": "plates", "garment": "garment", "relit": "relit",
    "comp": "comp_locked",
}


def work_root(root: str) -> str:
    return os.path.join(root, "data", "work")


def seg_dir(root: str, artifact: str, sid: int) -> str:
    """某标准件某段的目录,如 data/work/alpha/seg_3。"""
    return os.path.join(work_root(root), _ARTIFACT_DIRS.get(artifact, artifact), f"seg_{sid}")


def artifact_path(root: str, artifact: str) -> str:
    """非分段标准件的路径(beats/final 等)。"""
    if artifact == "beats":
        return os.path.join(work_root(root), "beats.json")
    if artifact == "final":
        return os.path.join(root, "data", "output", "final.mp4")
    if artifact == "camera":
        return os.path.join(work_root(root), "camera")
    return os.path.join(work_root(root), _ARTIFACT_DIRS.get(artifact, artifact))


def frame_count(d: str) -> int:
    return len(glob.glob(os.path.join(d, "f*.png")))


def write_manifest(d: str, **fields) -> None:
    os.makedirs(d, exist_ok=True)
    fields.setdefault("schema", SCHEMA)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)


def read_manifest(d: str) -> dict | None:
    p = os.path.join(d, "manifest.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def artifact_status(root: str, segments: list[int]) -> dict:
    """供 WebUI:每个标准件每段的产出情况(帧数)。"""
    status: dict[str, dict] = {}
    for art in ["src", "locked", "alpha", "plates", "garment", "relit", "comp"]:
        status[art] = {sid: frame_count(seg_dir(root, art, sid)) for sid in segments}
    status["beats"] = os.path.isfile(artifact_path(root, "beats"))
    status["final"] = os.path.isfile(artifact_path(root, "final"))
    return status
