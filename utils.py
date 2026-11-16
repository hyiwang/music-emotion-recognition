import torch
from torch import nn
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from typing import Dict, List
from pathlib import Path

from config import *

class ToMono(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

class PadTrim(nn.Module):
    def __init__(self, start_pos, max_len):
        super().__init__()
        self.start_pos = start_pos
        self.max_len = max_len
        self.end_pos = start_pos + max_len

    def forward(self, waveform):
        # shape[0]是声道数
        pad_amount = self.end_pos - waveform.shape[1]
        if waveform.shape[1] < self.end_pos:
            waveform = nn.functional.pad(waveform, (0, pad_amount))

        waveform = waveform[:, self.start_pos : self.end_pos]
        return waveform

class NormalizeMel(nn.Module):
    """按 mel 频带做标准化：x' = (x - mean) / std。

    mean/std 是在训练集上统计出的、形状为 (n_mels,) 的常数。
    输入形状 (B, n_mels, T) 或 (1, n_mels, T)，输出形状不变。
    """

    def __init__(self, mean, std, eps=1e-6):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1))
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)

def build_transform(mel_stats=None, start_time=None, duration=None,
                    orig_freq=ORIGINAL_SAMPLE_RATE):
    """构造训练、测试、推理共用的音频预处理流水线。

    Args:
        mel_stats: (mean, std) 每个 mel 频带的统计量；None 表示不做标准化。
        start_time: 截取起点（秒），默认与训练一致（15.0s）。
        duration: 截取时长（秒），默认 30s。
        orig_freq: 输入音频的采样率。训练/测试假设 44.1kHz；
            推理时可传入文件真实采样率。
    """
    if start_time is None:
        start_time = LABEL_START_TIME
    if duration is None:
        duration = PADTRIM_TIME

    steps = [
        ToMono(),
        transforms.Resample(orig_freq=orig_freq, new_freq=SAMPLE_RATE),
        PadTrim(start_pos=int(SAMPLE_RATE * start_time),
                max_len=int(SAMPLE_RATE * duration)),
        transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                                  hop_length=HOP_LENGTH, n_mels=N_MELS),
        transforms.AmplitudeToDB(),
    ]
    if mel_stats is not None:
        steps.append(NormalizeMel(*mel_stats))
    return nn.Sequential(*steps)

@torch.no_grad()
def compute_mel_stats(audio_paths, base_transform):
    """遍历训练集，统计每个 mel 频带的均值和标准差。

    base_transform 必须是“不含 NormalizeMel”的流水线。
    返回 (mean, std)，形状均为 (n_mels,)。
    """
    total = torch.zeros(N_MELS, dtype=torch.float64)
    total_sq = torch.zeros(N_MELS, dtype=torch.float64)
    n_frames = 0

    for path in tqdm(audio_paths, desc="mel stats"):
        waveform, _ = torchaudio.load(path)
        mel = base_transform(waveform)          # (1, n_mels, T)
        total += mel.sum(dim=(0, 2)).double()
        total_sq += mel.double().pow(2).sum(dim=(0, 2))
        n_frames += mel.shape[-1]

    mean = total / n_frames
    std = (total_sq / n_frames - mean.pow(2)).clamp_min(1e-8).sqrt()
    return mean.float(), std.float()

def save_mel_stats(path, mean, std, start_time=LABEL_START_TIME, duration=PADTRIM_TIME):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std, "n_mels": N_MELS,
                "sample_rate": SAMPLE_RATE,
                "start_time": start_time, "duration": duration}, path)
    print(f"Saved mel stats to: {path}")

def load_mel_stats(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Mel stats not found: {path}. Run train.py first to generate it."
        )
    data = torch.load(path, map_location="cpu")
    expected = {"n_mels": N_MELS, "sample_rate": SAMPLE_RATE,
                "start_time": LABEL_START_TIME, "duration": PADTRIM_TIME}
    stale = {k: (data.get(k), v) for k, v in expected.items() if data.get(k) != v}
    if stale:
        detail = ", ".join(f"{k}: file={a}, expected={b}" for k, (a, b) in stale.items())
        raise ValueError(
            f"{path} does not match the current preprocessing ({detail}). "
            f"Delete it and rerun train.py."
        )
    return data["mean"], data["std"]

class DEAMDataset(Dataset):
    def __init__(self, audio_paths, labels, transform=None):
        self.audio_paths = audio_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        song_id = int(audio_path.stem)
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        label = self.labels[song_id]
        return waveform, label

    def __len__(self):
        return len(self.audio_paths)

def read_label_csv(arousal_path, valence_path):
    arousal_df = pd.read_csv(arousal_path, index_col='song_id')
    valence_df = pd.read_csv(valence_path, index_col='song_id')

    target_cols = [f'sample_{ms}ms' for ms in range(15000, 44500 + 1, 500)]

    labels = {}
    for song_id in valence_df.index:
        arousal = arousal_df.loc[song_id, target_cols].values.astype('float32')  # (60)
        valence = valence_df.loc[song_id, target_cols].values.astype('float32')  # (60)

        label = np.stack([arousal, valence], axis=1)  # (60, 2)
        labels[int(song_id)] = label

    return labels

def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def plot_line_graph(data):
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.show()

def plot_train_test_loss(results):
    """
    绘制训练误差和测试误差随 epoch 变化的曲线。

    Args:
        results: train() 返回的字典，包含 train_loss 和 test_loss。
    """
    train_losses = results["train_loss"]
    test_losses = results["test_loss"]

    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 训练误差
    axes[0].plot(epochs, train_losses, marker="o", label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_xticks(list(epochs))
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 测试误差
    axes[1].plot(epochs, test_losses, marker="o", color="orange", label="Test Loss")
    axes[1].set_title("Test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_xticks(list(epochs))
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.tight_layout()
    plt.show()

def plot_pred_label(x, pred, label=None):

    x = to_numpy(x)
    pred = to_numpy(pred)
    # 去掉 batch 维度
    if pred.ndim == 3:
        pred = pred[0]

    if label is not None:
        label = to_numpy(label)
        if label.ndim == 3:
            label = label[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    names = ["Arousal", "Valence"]

    for i, name in enumerate(names):
        axes[i].plot(x, pred[:, i], label="Prediction",
                     marker="o", markersize=3)

        if label is not None:
            label = np.asarray(label)
            axes[i].plot(x, label[:, i], label="Ground Truth",
                         marker="o", markersize=3)

        axes[i].set_title(name)
        axes[i].set_ylabel("Value")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend()

    axes[1].set_xlabel("Time")
    fig.tight_layout()
    plt.show()

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(dataloader)

    return train_loss

def test_step(model, dataloader, loss_fn, device):
    model.eval()

    test_loss = 0.0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

        test_loss /= len(dataloader)
        return test_loss

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int=5,
          device: torch.device="cpu",
          save_every: int=0) -> Dict[str, List]:
    results = {"train_loss": [], "test_loss": []}

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss = test_step(model, test_dataloader, loss_fn, device)

        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_model(model=model,
                       target_dir=MODEL_DIR,
                       model_name=f"rnn_epoch_{epoch + 1}.pth")

    return results

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir / model_name

    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)