import numpy as np
from scipy import signal
import librosa
from src.generic_utils import plot_spectogram

def generate_ft(audio, sr, verbose=False):
    n_fft = 512
    ft = np.abs(librosa.stft(audio, n_fft=n_fft))
    if verbose:
        plot_spectogram(ft)
    return ft

def generate_mfcc(audio, sample_rate, verbose=False):
    mfcc = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=20)
    if verbose:
        plot_spectogram(mfcc, db=True)
    return mfcc.T

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)