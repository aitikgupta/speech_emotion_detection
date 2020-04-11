import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  pass
from src.preprocess import generate_features
from src.features import log_specgram
from src.generic_utils import label_to_str
from keras.models import load_model
import numpy as np

def predict_emotion(dataset, labels=None, mode="dev", model_path="./model/model.h5", verbose=False):
  print("[INFO] Predicting!")
  feat = generate_features(dataset, datatype=mode)

  model = load_model(model_path)

  preds = model.predict(feat)
  
  if verbose:
    if mode == "user":  
      for pred in range(preds.shape[0]):
        print(f"Predicted: {label_to_str(np.argmax(preds[pred]))}")      
    else:
      for pred in range(preds.shape[0]):
        print(f"Actual: {label_to_str(labels[pred])}, Predicted: {label_to_str(np.argmax(preds[pred]))}")
  return np.argmax(preds, axis=1)