"""S1 卡点:从音乐里检测节拍,生成 beats.json,作为后续所有切段依据。

用法:
    python src/s1_beats.py                         # 用默认 config.yaml
    python src/s1_beats.py --config config.yaml --out data/work/beats.json
    python src/s1_beats.py --plot                   # 额外导出一张卡点可视化图(可选)

输出 data/work/beats.json:
    {
      "bpm": 120.0,
      "fps": 30,
      "audio_duration": 65.3,
      "source": ".../music.wav",
      "cut_times": [0.0, 0.5, 1.0, ...],            # 切换发生的时间点(秒)
      "segments": [                                  # 相邻切点之间为一段
        {"id": 0, "start_time": 0.0, "end_time": 0.5,
         "start_frame": 0, "end_frame": 15, "duration": 0.5}, ...
      ]
    }

config(beats 段)说明:
    detect: auto         # auto = librosa 自动卡点;manual = 仅用 override_seconds
    subdivide: 1         # 每 N 拍切一次(1=每拍,2=隔拍,降低切换频率)
    override_seconds: [] # 非空则作为卡点时间(秒),覆盖自动结果
    include_start_end: true  # 是否用 0 和音频结尾把首尾补成完整段(默认 true)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 允许以脚本方式直接运行(把项目根加入 import 路径)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import get, load_config, resolve_path  # noqa: E402


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK,会把中文日志打乱;尽量切到 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def detect_beats(audio_path: str) -> tuple[float, list[float], float]:
    """用 librosa 检测节拍。返回 (bpm, 节拍时间列表[秒], 音频时长[秒])。"""
    try:
        import librosa
    except ImportError as e:
        raise SystemExit(
            "缺少依赖 librosa,请先安装:pip install librosa soundfile"
        ) from e

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(tempo if not hasattr(tempo, "__len__") else tempo[0])
    return bpm, [float(t) for t in beat_times], duration


def build_cut_times(
    beat_times: list[float],
    duration: float,
    *,
    subdivide: int,
    override: list[float] | None,
    include_start_end: bool,
) -> list[float]:
    """由节拍/手动覆盖生成最终的切换时间点(去重、排序、限定在 [0, duration])。"""
    if override:
        cuts = [float(t) for t in override]
    else:
        step = max(1, int(subdivide))
        cuts = beat_times[::step]

    if include_start_end:
        cuts = [0.0, *cuts, duration]

    # 限定范围 + 去重(容差 1ms)+ 排序
    cuts = sorted(t for t in cuts if 0.0 <= t <= duration)
    deduped: list[float] = []
    for t in cuts:
        if not deduped or abs(t - deduped[-1]) > 1e-3:
            deduped.append(round(t, 3))
    return deduped


def cuts_to_segments(cut_times: list[float], fps: float) -> list[dict]:
    """把切点序列转成相邻段落,带秒与帧号。"""
    segments = []
    for i in range(len(cut_times) - 1):
        start, end = cut_times[i], cut_times[i + 1]
        segments.append(
            {
                "id": i,
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "start_frame": int(round(start * fps)),
                "end_frame": int(round(end * fps)),
                "duration": round(end - start, 3),
            }
        )
    return segments


def maybe_plot(audio_path: str, cut_times: list[float], out_png: str) -> None:
    """可选:导出卡点可视化波形图,便于人工核对/微调。"""
    try:
        import librosa
        import librosa.display  # noqa: F401
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] 跳过:需要 matplotlib(pip install matplotlib)")
        return

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    plt.figure(figsize=(14, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    for t in cut_times:
        plt.axvline(t, color="r", linestyle="--", linewidth=0.8)
    plt.title(f"卡点 ({len(cut_times)} 个切点)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[plot] 已保存 {out_png}")


def run(config_path: str | None, out_path: str | None, do_plot: bool) -> dict:
    _force_utf8_stdout()
    cfg = load_config(config_path)

    audio_path = resolve_path(cfg, get(cfg, "input.music", "data/input/music.wav"))
    if not os.path.isfile(audio_path):
        raise SystemExit(f"找不到音乐文件:{audio_path}(请放入 data/input/)")

    fps = float(get(cfg, "project.fps", 30))
    detect_mode = get(cfg, "beats.detect", "auto")
    subdivide = int(get(cfg, "beats.subdivide", 1))
    override = get(cfg, "beats.override_seconds", []) or []
    include_start_end = bool(get(cfg, "beats.include_start_end", True))

    if detect_mode == "manual" and not override:
        raise SystemExit("beats.detect=manual 但 override_seconds 为空,无法卡点")

    if override:
        print(f"[s1] 使用手动卡点 override_seconds({len(override)} 点)")
        bpm, beat_times, duration = 0.0, [], float(max(override) + 1.0)
        # 手动模式下仍尝试取真实时长(用于补尾段)
        try:
            import librosa

            duration = float(librosa.get_duration(path=audio_path))
        except Exception:
            pass
    else:
        print(f"[s1] librosa 检测节拍:{audio_path}")
        bpm, beat_times, duration = detect_beats(audio_path)
        print(f"[s1] BPM≈{bpm:.1f},检测到 {len(beat_times)} 拍,时长 {duration:.2f}s")
        if len(beat_times) < 2:
            print(
                "[s1][警告] 检测到的节拍过少,可能音乐节奏不明显或音频异常;"
                "可改用 beats.detect=manual + override_seconds 手动指定卡点。"
            )

    cut_times = build_cut_times(
        beat_times,
        duration,
        subdivide=subdivide,
        override=override,
        include_start_end=include_start_end,
    )
    segments = cuts_to_segments(cut_times, fps)

    result = {
        "bpm": round(bpm, 2),
        "fps": fps,
        "audio_duration": round(duration, 3),
        "source": audio_path,
        "cut_times": cut_times,
        "segments": segments,
    }

    if out_path is None:
        out_path = resolve_path(cfg, "data/work/beats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[s1] 已写出 {len(segments)} 段 → {out_path}")

    if do_plot:
        maybe_plot(audio_path, cut_times, os.path.splitext(out_path)[0] + ".png")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="S1 卡点:音乐节拍 → beats.json")
    ap.add_argument("--config", default=None, help="配置文件路径(默认项目根 config.yaml)")
    ap.add_argument("--out", default=None, help="输出 json 路径(默认 data/work/beats.json)")
    ap.add_argument("--plot", action="store_true", help="额外导出卡点可视化图")
    args = ap.parse_args()
    run(args.config, args.out, args.plot)


if __name__ == "__main__":
    main()
