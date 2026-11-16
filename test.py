import argparse
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from model import RNN
from utils import build_transform, load_mel_stats, read_label_csv
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1, help="测试歌曲数量，1～5")
args = parser.parse_args()

if not 1 <= args.n <= 5:
    parser.error("--n 必须在 1～5 之间")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_dir = Path("data/DEAM/audio")
model_path = Path("models/rnn.pth")

labels = read_label_csv(
    Path("data/DEAM/annotations/annotations_average/dynamic/arousal.csv"),
    Path("data/DEAM/annotations/annotations_average/dynamic/valence.csv")
)

audio_paths = [
    p for p in audio_dir.glob("*.mp3")
    if p.stem.isdigit() and int(p.stem) in labels
]

if len(audio_paths) < args.n:
    raise ValueError(f"可用歌曲只有 {len(audio_paths)} 首")

selected = random.sample(audio_paths, args.n)

mel_mean, mel_std = load_mel_stats(MEL_STATS_PATH)
transform = build_transform((mel_mean, mel_std))

model = RNN(n_mels=N_MELS).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 标签对应 15.0～44.5 秒，每隔 0.5 秒一个点
times = np.arange(15.0, 45.0, 0.5)

results = []

with torch.inference_mode():
    for path in selected:
        waveform = transform(torchaudio.load(path)[0]).unsqueeze(0).to(device)

        pred = model(waveform).squeeze(0).cpu().numpy()
        label = np.asarray(labels[int(path.stem)])

        results.append((path.stem, pred, label))

# 所有歌曲画在同一张图中
fig, axes = plt.subplots(
    args.n, 2,
    figsize=(12, 3 * args.n),
    squeeze=False,
    sharex=True
)

for row, (song_id, pred, label) in enumerate(results):
    for col, name in enumerate(["Arousal", "Valence"]):
        ax = axes[row, col]
        ax.plot(times, pred[:, col], label="Prediction")
        ax.plot(times, label[:, col], "--", label="Ground Truth")
        ax.set_title(f"Song {song_id} - {name}")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        if row == args.n - 1:
            ax.set_xlabel("Time (seconds)")

fig.tight_layout()
# Path("results").mkdir(exist_ok=True)
# fig.savefig(f"results/test_{args.n}_songs.png", dpi=150)
plt.show()
