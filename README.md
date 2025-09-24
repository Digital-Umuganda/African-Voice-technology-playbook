# African-Voice-technology-playbook

## Load model directly, WHISPER
```
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("jq/whisper-large-v3-kin-track-b")
model = AutoModelForSpeechSeq2Seq.from_pretrained("jq/whisper-large-v3-kin-track-b")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

sample, sample_rate = torchaudio.load("path/to/audio.wav")

result = pipe(sample)
print(result["text"])
```

## wav2vec2-bert

```
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC
import torch
import torchaudio

processor = Wav2Vec2BertProcessor.from_pretrained("badrex/w2v-bert-2.0-kinyarwanda-asr")
model = Wav2Vec2BertForCTC.from_pretrained("badrex/w2v-bert-2.0-kinyarwanda-asr")

audio_input, sample_rate = torchaudio.load("path/to/audio.wav")

# preprocess
inputs = processor(audio_input.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

# inference
with torch.no_grad():
    logits = model(**inputs).logits

# decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print(transcription)
```
## MMS
```
import soundfile as sf
import torch

from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
processor.tokenizer.set_target_lang("kin")
model.load_adapter("kin")

sample, sample_rate = torchaudio.load("path/to/audio.wav")

inputs = processor(sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# 'joe keton disapproved of films and buster also had reservations about the media'

```
## nemo models
```
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")


asr_model.transcribe(['<audio_file_path>'])

```
