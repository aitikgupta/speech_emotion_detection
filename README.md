# Speech Emotion Detection built using Conv1D+LSTM layers
This is a user-driven speech emotion detection system.
#### Tech Stack:
*   TensorFlow
*   libROSA
*   PyAudio

## Steps to reproduce:
Note: To maintain the ennvironments, I highly recommend using [conda](https://anaconda.org/).

```
git clone https://github.com/aitikgupta/speech_emotion_detection.git
cd speech_emotion_detection
conda env create -f environment.yml
conda activate {environment name, for eg. conda activate kaggle}
python main.py
```
There are 3 things which can be done:
1. Train the model again (This will take time, on a GeForce GTX 1650 GPU, training took around 1 hour)
2. Randomly select sample voices and test the model on actual and predicted values
3. Test the model on your own voice sample

## Classes:
*   "male_calm"
*   "male_happy"
*   "male_sad"
*   "male_dislike"
*   "male_fearful"
*   "male_surprised"
*   "female_calm"
*   "female_happy"
*   "female_sad"
*   "female_dislike"
*   "female_fearful"
*   "female_surprised"

## Inspiration:
There has been a wide variety of convolutional models on the internet, for speech emotion detection.
#### However, in a time-dependent data, just using convolutions doesn't account for the correlations within time intervals.
Therefore, using RNNs with Convolutions, the model can be much more robust in understanding the intent of the user.

## Pipeline:
1. The input audio is first converted into a spectrogram, which uses [Fourier Transformation](https://en.wikipedia.org/wiki/Fourier_transform) to convert time domain into frequency domain.
2. The scales are shifted into log, and can also be converted into the [Mel scale.](https://en.wikipedia.org/wiki/Mel_scale)
3. For every fold in training and validation split (using [Stratified K Fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)):
##### Random processing techniques are applied to voices such as shifting, pitch tuning, adding white noise, etc. to make the model more robust. [This was key to make it usable in realtime.]
4. The features generated are thus fed into Conv1D layers, and ultimately LSTMs.
5. The last Dense layer contains 12 units with a [Softmax](https://en.wikipedia.org/wiki/Softmax_function) activation.

## Limitations:
*   "Emotions" are hard to annotate, even for humans. There is no "Perfect" dataset for such problem.
*   [Here's a great article](https://towardsdatascience.com/whats-wrong-with-spectrograms-and-cnns-for-audio-processing-311377d7ccd) as to why convolutions and spectrograms are not "ideal" for audio processing.
