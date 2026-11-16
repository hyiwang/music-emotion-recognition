import argparse
import numpy as np
import torch
import torchaudio

from model import RNN
from utils import build_transform, load_mel_stats, plot_pred_label
from config import *

START_TIME = LABEL_START_TIME

parser = argparse.ArgumentParser(description="Predict using a model.")
parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio for prediction.")
parser.add_argument("--model_path", type=str, default="models/rnn.pth", help="Path to the model file.")
args = parser.parse_args()

audio_path = args.audio_path
model_path = args.model_path

device = "cuda" if torch.cuda.is_available() else "cpu"

waveform, original_sample_rate = torchaudio.load(audio_path)

mel_mean, mel_std = load_mel_stats(MEL_STATS_PATH)
transform = build_transform((mel_mean, mel_std), start_time=START_TIME,
                            orig_freq=original_sample_rate)

audio = transform(waveform).to(device)

model = RNN().to(device)
model.load_state_dict(torch.load(model_path))

times = np.arange(START_TIME, START_TIME + PADTRIM_TIME - 1e-6, 0.5)

model.eval()
with torch.inference_mode():
    y_pred = model(audio)
    plot_pred_label(times, y_pred)