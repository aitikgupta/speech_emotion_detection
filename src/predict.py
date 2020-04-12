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
  
  if mode == "user":  
    predicts = []
    probabilities = []
    for pred in range(preds.shape[0]):
      value = np.argmax(preds[pred])
      confidence = max(preds[pred])
      predicts.append(value)
      probabilities.append(confidence)
      if verbose:
        print(f"Predicted: {label_to_str(value)}\tConfidence: {confidence}")
    return np.array(predicts), np.array(probabilities)
  else:
    actual = []
    predicts = []
    probabilities = []
    for pred in range(preds.shape[0]):
      true_val = labels[pred]
      pred_val = np.argmax(preds[pred])
      confidence = max(preds[pred])
      actual.append(true_val)
      predicts.append(pred_val)
      probabilities.append(confidence)
      if verbose:
        print(f"Actual: {label_to_str(true_val)}\tPredicted: {label_to_str(pred_val)}\tConfidence: {confidence}")
    return np.array(actual), np.array(predicts), np.array(probabilities)