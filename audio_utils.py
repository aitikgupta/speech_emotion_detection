import pyaudio
import wave
import librosa
import numpy as np
from ctypes import *

def normalize_audio(data):
    return librosa.util.normalize(data)

def load_audio(path=None, normalize=True, input_length=70000):
    data, sample_rate = librosa.load(path, offset=0.2)
    if normalize:
        data = normalize_audio(data)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data, sample_rate

def save_audio(data, path="output", sample_rate=22050, normalize=True):
    librosa.output.write_wav(path, data, sr=sample_rate, norm=normalize)

def py_error_handler(filename, line, function, err, fmt):
        pass
def play_audio(path="output/recording.wav", verbose=True):
    if verbose:
        print("[INFO] Playing Audio!")
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)

    
    chunk = 1024    
    f = wave.open(path, "rb")
    p = pyaudio.PyAudio()  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    data = f.readframes(chunk)
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)
    stream.stop_stream()  
    stream.close()
    p.terminate()
    asound.snd_lib_error_set_handler(None)

def record_audio(output_path="output/recording.wav", chunk=1024, channels=1, rate=48000, duration=3.35):
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)

    FORMAT = pyaudio.paInt16
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []

    print("[INFO] Recording!")

    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("[INFO] Done Recording!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    asound.snd_lib_error_set_handler(None)


def add_noise(data):
    noise_amp = 0.085*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
    
def add_shift(data):
    s_range = int(np.random.uniform(low=-10, high = 10)*1000)
    return np.roll(data, s_range)
        
def change_pitch(data, sample_rate=22050):
    pitch_pm = 2.5
    strength = np.random.uniform(-1,1)
    pitch_change =  pitch_pm * strength
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change)
    return data
    
def change_speed_pitch(data):
    audio = data.copy()
    length_change = np.random.uniform(low=0.85, high = 1)
    speed_fac = 1.1  / length_change
    tmp = np.interp(np.arange(0,len(audio),speed_fac),np.arange(0,len(audio)),audio)
    minlen = min(audio.shape[0], tmp.shape[0])
    audio *= 0
    audio[0:minlen] = tmp[0:minlen]
    return audio