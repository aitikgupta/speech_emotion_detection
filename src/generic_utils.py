import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def give_emotion(value):
    if value == 1 or value == 2:
        return 0
    elif value == 3:
        return 1
    elif value == 4:
        return 2
    elif value == 5 or value == 7:
        return 3
    elif value == 6:
        return 4
    else:
        return 5

def give_label(filename):
    nm = filename.split(".")[0].split("-")
    gender = int(nm[-1])
    emotion = give_emotion(int(nm[2]))
    if gender%2: 
        return emotion
    else:
        return emotion+6

def label_to_str(label):
    if label == 0:
        lab = "male_calm"
    elif label == 1:
        lab = "male_happy"
    elif label == 2:
        lab = "male_sad"
    elif label == 3:
        lab = "male_dislike"
    elif label == 4:
        lab = "male_fearful"
    elif label == 5:
        lab = "male_surprised"
    elif label == 6:
        lab = "female_calm"
    elif label == 7:
        lab = "female_happy"
    elif label == 8:
        lab = "female_sad"
    elif label == 9:
        lab = "female_dislike"
    elif label == 10:
        lab = "female_fearful"
    elif label == 11:
        lab = "female_surprised"
    return lab

def plot_spectogram(D, db=False):
    if db:
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.min), y_axis='log', x_axis='time')
    else:
        librosa.display.specshow(D, y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def plot_traintest(history):
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    for fold in range(len(history.keys())):
        x1 += history[fold]["loss"]
        x2 += history[fold]["val_loss"]
        x3 += history[fold]["acc"]
        x4 += history[fold]["val_acc"]
    plt.xlabel("Epochs")
    plt.plot(x1, label="Training Loss")
    plt.plot(x2, label="Validation Loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.show()
    plt.plot(x3, label="Training Accuracy")
    plt.plot(x4, label="Validation Accuracy")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.show()

def generate_report(ytrue, ypred, verbose=True, just_acc=False):
    if verbose:
        print("[INFO] Generating Report!")
    targets = ["male_calm","male_happy","male_sad","male_dislike","male_fearful","male_surprised","female_calm","female_happy","female_sad","female_dislike","female_fearful","female_surprised"]
    if just_acc:
        print("Total Accuracy:", accuracy_score(y_true=ytrue, y_pred=ypred))
        return
    print(classification_report(y_true=ytrue, y_pred=ypred, target_names=targets, labels=range(len(targets)), zero_division=0))
