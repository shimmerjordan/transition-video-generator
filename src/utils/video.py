"""视频 / 帧 I/O 与 ffmpeg 工具,供 s2..s8 共用。

- 帧读写用 OpenCV(BGR np.uint8);路径含非 ASCII 时用 imdecode/imencode 兜底。
- 编码与音频混流走 imageio-ffmpeg 自带的 ffmpeg 二进制,无需系统安装 ffmpeg。
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Iterable, Iterator

import cv2
import numpy as np


# ---------- ffmpeg ----------

def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise SystemExit("找不到 ffmpeg,请安装:pip install imageio-ffmpeg") from e


# ---------- 路径安全的帧读写(兼容中文路径)----------

def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"编码失败:{path}")
    buf.tofile(path)


def ensure_dir(d: str) -> None:
    if d:
        os.makedirs(d, exist_ok=True)


# ---------- 视频信息与读帧 ----------

def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频:{path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
        "count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def read_frames(path: str, start: int = 0, count: int | None = None) -> Iterator[np.ndarray]:
    """逐帧读取 [start, start+count) 的 BGR 帧。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频:{path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n = 0
    while count is None or n < count:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        n += 1
    cap.release()


# ---------- 写视频(从帧迭代器,经 ffmpeg 编码)----------

class FrameWriter:
    """把 BGR 帧逐帧管道给 ffmpeg 编码为 H.264 mp4(无音轨)。"""

    def __init__(self, out_path: str, fps: float, size: tuple[int, int], crf: int = 17):
        ensure_dir(os.path.dirname(out_path))
        w, h = size
        self.size = size
        cmd = [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
            out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_video(frames: Iterable[np.ndarray], out_path: str, fps: float,
                size: tuple[int, int], crf: int = 17) -> None:
    with FrameWriter(out_path, fps, size, crf) as w:
        for f in frames:
            w.write(f)


# ---------- 音频混流 / 拼接 ----------

def mux_audio(video_path: str, audio_path: str, out_path: str,
              shortest: bool = True) -> None:
    """给无声视频加上音轨(视频流直接 copy)。"""
    ensure_dir(os.path.dirname(out_path))
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-i", video_path, "-i", audio_path,
           "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0"]
    if shortest:
        cmd.append("-shortest")
    cmd.append(out_path)
    subprocess.run(cmd, check=True)


def extract_audio(src_video: str, out_audio: str) -> bool:
    """从视频抽音轨;无音轨返回 False。"""
    ensure_dir(os.path.dirname(out_audio))
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src_video,
           "-vn", "-acodec", "pcm_s16le", out_audio]
    return subprocess.run(cmd).returncode == 0 and os.path.isfile(out_audio)


# ---------- 变换矩阵序列读写(s2 ↔ s8)----------

def save_transforms(path: str, transforms: list, meta: dict | None = None) -> None:
    """transforms: 每帧 2x3 仿射矩阵(list)。"""
    ensure_dir(os.path.dirname(path))
    payload = {"meta": meta or {}, "transforms": [np.asarray(t).tolist() for t in transforms]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_transforms(path: str) -> tuple[list[np.ndarray], dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mats = [np.asarray(t, dtype=np.float64) for t in payload["transforms"]]
    return mats, payload.get("meta", {})
