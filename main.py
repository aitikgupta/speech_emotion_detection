import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from src.dataload import load_data
from src.train import train_model
from src.predict import predict_emotion
from src.audio_utils import record_audio, play_audio
print("Welcome to my Speech-Emotion-Detection project. Check out my handles:-\n\n[linkedin.com/in/aitik-gupta][github.com/aitikgupta][kaggle.com/aitikgupta]\n\n")

DATASET_PATH = "./voices"
OUTPUT_PATH = "./output"
MODEL_PATH = "./model/model.h5"

choice = int(input("1) Train the model again.\n2) Test the model on random voices.\n3) Test the model by your voice.\nEnter choice: "))
if choice == 1:
    print("[INFO] Model file will be overwritten!")
    dataset, labels = load_data(DATASET_PATH, mode="dev", n_random=-1, play_runtime=False)
    train_model(dataset=dataset, labels=labels, model_path=MODEL_PATH, n_splits=5, learning_rate=0.0001, epochs=30, batch_size=64, verbose=True)
elif choice == 2:
    dataset, labels = load_data(DATASET_PATH, mode="dev", n_random=3, play_runtime=True)
    _ = predict_emotion(dataset, labels, mode="dev", model_path=MODEL_PATH, verbose=True)
else:
    recording_path = os.path.join(OUTPUT_PATH, "recording.wav")
    inp = str(input(f"Record audio again? [Default {recording_path} will be used] (y|n): ")).lower()
    if inp == "y" or inp == "yes":
        record_audio(output_path=recording_path)
    dataset, _ = load_data(OUTPUT_PATH, mode="user", n_random=-1, play_runtime=True)
    _ = predict_emotion(dataset, mode="user", model_path=MODEL_PATH, verbose=True)
