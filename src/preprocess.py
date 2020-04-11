import numpy as np
from src.audio_utils import load_audio, add_noise, add_shift, change_pitch, change_speed_pitch
from src.features import log_specgram

def scale(inp):
    return (inp - inp.mean())/inp.std()

def extract_features(batch, batch_path):
    features = []
    for data, input_path in zip(batch, batch_path):
        if input_path == 0:
            _, _, spec = log_specgram(data, sample_rate=22050)
        elif input_path == 1:
            _, _, spec = log_specgram(add_noise(data), sample_rate=22050)
        elif input_path == 2:
            _, _, spec = log_specgram(add_shift(data), sample_rate=22050)
        elif input_path == 3:
            _, _, spec = log_specgram(change_pitch(data), sample_rate=22050)
        else:
            _, _, spec = log_specgram(change_speed_pitch(data), sample_rate=22050)
        features.append(scale(spec))
    return np.array(features)

def generate_features(dataset, datatype="train"):
    if datatype == "train":
        paths  = np.random.choice([0,1,2,3,4], size = len(dataset))
    else:
        paths = np.random.choice([0], size = len(dataset))
    features = extract_features(dataset, paths)
    return np.array(features)        
