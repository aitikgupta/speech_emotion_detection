import os
import numpy as np
from src.audio_utils import load_audio, play_audio
from src.generic_utils import give_label

def load_data(path="./voices", mode="dev", n_random=-1, play_runtime=False, verbose=True):
    if verbose:
        print("[INFO] Loading Data!")
    dataset = []
    labels = []
    if mode == "user":
        filenames = os.listdir(path)
        for i, voice in enumerate(filenames):
            if play_runtime:
                play_audio(os.path.join(path, voice))
            audio, sample_rate = load_audio(os.path.join(path, voice), normalize=True)
            dataset.append(audio)
    else:    
        if n_random == -1:
            filenames = os.listdir(path)
        else:
            filenames = np.random.choice(os.listdir(path), n_random)
        for i, voice in enumerate(filenames):
            if play_runtime:
                play_audio(os.path.join(path, voice))
            audio, sample_rate = load_audio(os.path.join(path, voice), normalize=True)
            dataset.append(audio)
            label = give_label(voice)
            labels.append(label)
    return np.array(dataset), np.array(labels)
