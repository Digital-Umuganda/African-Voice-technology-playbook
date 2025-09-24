
## Model Description

<!-- Provide a longer summary of what this model is. -->
This model is an end-to-end deep-learning-based Kinyarwanda Text-to-Speech (TTS). The model was trained using the Coqui's TTS library,  and the YourTTS[1] architecture.


# Usage

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
Install the Coqui's TTS library:
```
pip install TTS
```
Download the files from this repo, then run: 

```
tts --text "text" --model_path model.pth --config_path config.json --speakers_file_path speakers.pth --speaker_wav conditioning_audio.wav --out_path out.wav
```
Where the conditioning audio is a wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder, you can give multiple file paths. The d_vectors is computed as their average.
