import argparse
import torch
from torch import nn
import torchaudio
from torchaudio import transforms

from model import RNN
from utils import ToMono, PadTrim

SAMPLE_RATE = 16000
START_TIME = 0 # seconds
PADTRIM_TIME = 30 # seconds
N_MELS = 64

parser = argparse.ArgumentParser(description="Predict using a model.")
parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio for prediction.")
parser.add_argument("--model_path", type=str, default="models/rnn.pth", help="Path to the model file.")
args = parser.parse_args()

audio_path = args.audio_path
model_path = args.model_path

device = "cuda" if torch.cuda.is_available() else "cpu"

waveform, original_sample_rate = torchaudio.load(audio_path)

transform = nn.Sequential(
    ToMono(),
    transforms.Resample(orig_freq=original_sample_rate, new_freq=SAMPLE_RATE),
    PadTrim(start_pos=int(SAMPLE_RATE * START_TIME), max_len=int(SAMPLE_RATE * PADTRIM_TIME)),
    transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS
    ),
    transforms.AmplitudeToDB()
)

audio = transform(waveform).to(device)

model = RNN().to(device)
model.load_state_dict(torch.load(model_path))

model.eval()
with torch.inference_mode():
    y_pred = model(audio)
    print(y_pred)