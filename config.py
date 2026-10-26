
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_DIR / "data" / "DEAM" / "audio"
ANNOTATION_DIR = (PROJECT_DIR / "data" / "DEAM" / "annotations" / "annotations_average" / "dynamic")
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

AROUSAL_PATH = ANNOTATION_DIR / "arousal.csv"
VALENCE_PATH = ANNOTATION_DIR / "valence.csv"
MODEL_PATH = MODEL_DIR / "rnn.pth"

# ==================== 音频处理参数 ====================
ORIGINAL_SAMPLE_RATE = 44100
SAMPLE_RATE = 16000

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

PADTRIM_TIME = 30 # seconds
LABEL_START_TIME = 15 # seconds
LABEL_INTERVAL = 0.5 # seconds
