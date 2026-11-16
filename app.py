"""Gradio demo：上传音乐 → 预测第 15–45 秒的 Arousal / Valence 曲线。

用法：
    python app.py                 # http://127.0.0.1:7860
    python app.py --port 7861 --share
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # 服务器环境不要用交互式后端
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import gradio as gr

from model import RNN
from utils import build_transform, load_mel_stats
from config import *

START_TIME = LABEL_START_TIME            # 15.0s
DURATION = PADTRIM_TIME                  # 30s
TIMES = np.arange(START_TIME, START_TIME + DURATION - 1e-6, LABEL_INTERVAL)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_stats():
    """读取逐频带统计量；缓存来自旧窗口时给出提示但仍可运行。"""
    try:
        return load_mel_stats(MEL_STATS_PATH)
    except ValueError as exc:
        print(f"[warn] {exc}")
        print("[warn] 继续使用这份缓存；重新训练后会自动刷新。")
        data = torch.load(MEL_STATS_PATH, map_location="cpu")
        return data["mean"], data["std"]


MEL_MEAN, MEL_STD = _load_stats()

if not Path(MODEL_PATH).exists():
    raise SystemExit(f"模型文件不存在：{MODEL_PATH}，请先运行 python train.py 训练模型。")

MODEL = RNN(n_mels=N_MELS).to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.eval()


def quadrant(arousal, valence):
    if arousal >= 0 and valence >= 0:
        return "正唤醒 · 正效价（兴奋、愉快）"
    if arousal >= 0 and valence < 0:
        return "正唤醒 · 负效价（紧张、愤怒）"
    if arousal < 0 and valence >= 0:
        return "负唤醒 · 正效价（平静、放松）"
    return "负唤醒 · 负效价（忧郁、悲伤）"


def infer(audio_path):
    if not audio_path:
        raise gr.Error("请先上传一段音频。")

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as exc:
        raise gr.Error(f"无法解码该音频：{exc}。请换 mp3 / wav 再试。")

    duration = waveform.shape[1] / sample_rate
    if duration < START_TIME:
        raise gr.Error(
            f"音频只有 {duration:.1f}s，模型分析的是第 {START_TIME:.0f}–"
            f"{START_TIME + DURATION:.0f}s，请上传更长的音频。"
        )

    transform = build_transform(
        (MEL_MEAN, MEL_STD), start_time=START_TIME, orig_freq=sample_rate
    )
    x = transform(waveform).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        pred = MODEL(x).squeeze(0).cpu().numpy()      # (60, 2)

    arousal = float(pred[:, 0].mean())
    valence = float(pred[:, 1].mean())

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(TIMES, pred[:, 0], color="#e4572e", lw=1.8, label="Arousal")
    ax.plot(TIMES, pred[:, 1], color="#2e86de", lw=1.8, label="Valence")
    ax.axhline(0.0, color="#888888", lw=0.8, ls="--")
    ax.set_xlim(START_TIME, START_TIME + DURATION)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted value")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()

    summary = (
        f"### 整体情绪估计（第 {START_TIME:.1f}–{START_TIME + DURATION:.1f} 秒）\n\n"
        "| 维度 | 均值 |\n|---|---|\n"
        f"| Arousal 唤醒度 | {arousal:+.2f} |\n"
        f"| Valence 效价 | {valence:+.2f} |\n\n"
        f"**情绪象限**：{quadrant(arousal, valence)}"
    )
    if duration < START_TIME + DURATION:
        summary += (
            f"\n\n> ⚠️ 音频只有 {duration:.1f}s，"
            f"{duration:.1f}–{START_TIME + DURATION:.0f}s 用静音补齐，结果仅供参考。"
        )
    return fig, summary


demo = gr.Interface(
    fn=infer,
    inputs=gr.Audio(type="filepath", label="上传音频（建议 ≥ 45 秒）", sources=["upload"]),
    outputs=[gr.Plot(label="预测曲线"), gr.Markdown(label="整体情绪")],
    title="音乐情感识别 · DEAM Arousal / Valence ♪",
    description=(
        "上传一段音乐，模型分析其 **第 15–45 秒**，输出 60 个时间点的唤醒度（Arousal）"
        "与效价（Valence）预测，取值约 −1 ~ 1。\n\n"
        "训练数据：DEAM；模型：2 层 BiGRU + 逐频带标准化。"
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Gradio demo.")
    parser.add_argument("--host", default=None, help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=7860, help="端口，默认 7860")
    parser.add_argument("--share", action="store_true", help="生成临时公网链接")
    args = parser.parse_args()

    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
