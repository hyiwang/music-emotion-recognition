import torch
from torch import nn
import torchaudio
from torchaudio import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse
from pathlib import Path

from model import RNN
from utils import ToMono, PadTrim, DEAMDataset
from utils import train, save_model, read_label_csv, plot_line_graph, plot_train_test_loss
from config import *


parser = argparse.ArgumentParser(description="Train model.")

parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the optimizer.")
parser.add_argument("--resume", type=str, default=None, help="Path to a pretrained state_dict (.pth) to resume training.")

args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

labels = read_label_csv(AROUSAL_PATH, VALENCE_PATH)

audio_paths = list(AUDIO_DIR.glob("*.mp3"))
n = len(audio_paths)
random.shuffle(audio_paths)

train_size = int(n * 0.8)
train_audio_paths = audio_paths[:train_size]
test_audio_paths = audio_paths[train_size:]

data_transform = nn.Sequential(
    ToMono(),
    transforms.Resample(orig_freq=ORIGINAL_SAMPLE_RATE, new_freq=SAMPLE_RATE),
    PadTrim(start_pos=int(SAMPLE_RATE * (LABEL_START_TIME - LABEL_INTERVAL)), max_len=int(SAMPLE_RATE * PADTRIM_TIME)),
    transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS
    ),
    transforms.AmplitudeToDB()
)

train_dataset = DEAMDataset(audio_paths=train_audio_paths, labels=labels, transform=data_transform)
test_dataset = DEAMDataset(audio_paths=test_audio_paths, labels=labels, transform=data_transform)
print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Train dataloader size: {len(train_dataloader)}, Test dataloader size: {len(test_dataloader)}")

RNN_model = RNN(n_mels=N_MELS, hidden_size=128, num_layers=2, num_labels=60, output_dim=2).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(RNN_model.parameters(), lr=LEARNING_RATE)

if args.resume is not None:
    resume_path = Path(args.resume)
    if not resume_path.exists():
        raise FileNotFoundError(f"Pretrained state_dict not found: {resume_path}")

    print(f"Loading state_dict from: {resume_path}")
    state_dict = torch.load(resume_path, map_location=device)
    RNN_model.load_state_dict(state_dict)
    print(f"Loaded successfully.")

results = train(model=RNN_model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=EPOCHS,
      device=device,
      save_every=10)

plot_train_test_loss(results)

save_model(model=RNN_model,
           target_dir="models",
           model_name="rnn.pth")
