
## Model Description

<!-- Provide a longer summary of what this model is. -->
This model is an end-to-end deep-learning-based Kinyarwanda Text-to-Speech (TTS). The model was trained using the Coqui's TTS library,  and the YourTTS[1] architecture.


# Usage

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
Install the Coqui's TTS library:
```
pip install TTS
```

## YourTTS Kinyarwanda
Download the files from the model's repo: 
```
from huggingface_hub import snapshot_download

model_path = "DigitalUmuganda/KinyarwandaTTS_female_voice"

snapshot_download(repo_id=model_path, local_dir=".")
```
And then:
```
tts --text "text" --model_path model.pth --config_path config.json --speakers_file_path speakers.pth --speaker_wav conditioning_audio.wav --out_path out.wav


```
## XTTS_V2 Kinyarwanda
Download the files from the model's repo: 
```
from huggingface_hub import snapshot_download

model_path = "DigitalUmuganda/xtts_based_male_female_health_model"

snapshot_download(repo_id=model_path, local_dir=".")
```
Run the model:
```
tts --text "mwiriwe neza, kwa kabiri hazaba inama." --model_path . --config_path config.json --speaker_wav male_conditioning_audio.wav --language_idx en --out_path out.wav
```
Where the conditioning audio is a wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder, you can give multiple file paths. The d_vectors is computed as their average.
