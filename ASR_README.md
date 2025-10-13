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

## Emformer streaming model
```
import sentencepiece as spm
import torch
from typing import List
from torchaudio.io import StreamReader


def post_process_hypos(tokens: List[int]) -> str:

    hypotheses = [x for x in tokens if x > 0 and x < 128]
    pred_texts = sp.decode(hypotheses)

    return pred_texts

bundle = torch.jit.load("scripted_wrapper.pt")
tokenizer_model_path=""
sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
```
#### Non-streaming predictions
```
tokens = bundle.predict(input_tensor)[0]
pred_text = post_process_hypos(tokens)

```
#### Streaming prediction section
```
class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]
        return chunk_with_context



hop_length=160
segment_length=16
right_context_length=4

sample_rate = 16000
segment_length = segment_length * hop_length
context_length = right_context_length * hop_length

print(f"Sample rate: {sample_rate}")
print(f"Main segment: {segment_length} frames ({segment_length / sample_rate} seconds)")
print(f"Right context: {context_length} frames ({context_length / sample_rate} seconds)")
src = "https://www.laits.utexas.edu/phonology/sounds/MP3/142b.mp3"

streamer = StreamReader(src)
streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate)

cacher = ContextCacher(segment_length, context_length)

state, hypothesis = None, None

stream_iterator = streamer.stream()


@torch.inference_mode()
def run_inference(num_iter=100):
    global state, hypothesis
    chunks = []
    feats = []
    # print("stream_iterator.length: ",len(list(stream_iterator)))
    for i, (chunk,) in enumerate(stream_iterator, start=1):
        first = (state is None)
        # print("chunk.shape: ",chunk.shape)
        segment = cacher(chunk[:, 0])

        hypos, state = wrapper.stream(segment, state, hypothesis)

        hypothesis = hypos

        print('Hypos:', post_process_hypos(hypos[0][0]), flush=True)

```
