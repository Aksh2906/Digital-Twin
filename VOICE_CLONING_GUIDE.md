# Voice Cloning: Feynman's Voice from a 10 Second Clip

XTTS v2 supports zero shot voice cloning. You give it one short audio clip of someone speaking and it clones that voice for any new text. No training, no GPU hours, no dataset preparation.

## Quick Start (15 minutes)

### Step 1: Get a Feynman audio clip

You need one clean 10 to 15 second WAV file of Feynman speaking. No background music, no other speakers, just his voice.

Easiest way: go to YouTube, find a Feynman clip, download it, trim to 10 seconds.

```bash
pip install yt-dlp pydub

# Download audio from a Feynman video
yt-dlp -x --audio-format wav -o "feynman_raw.wav" "https://www.youtube.com/watch?v=P1ww1IXRfTA"
```

Then trim it to a clean 10 second section using Python:

```python
from pydub import AudioSegment

audio = AudioSegment.from_wav("feynman_raw.wav")

# Pick a section where Feynman is speaking clearly
# Change these numbers based on where in the clip he speaks
start_ms = 5000   # start at 5 seconds
end_ms = 15000    # end at 15 seconds

clip = audio[start_ms:end_ms]
clip = clip.set_channels(1).set_frame_rate(22050)
clip.export("data/feynman_voice_sample.wav", format="wav")
print("Saved 10 second clip")
```

Or just manually trim any Feynman audio to 10 seconds using any audio editor (Audacity, QuickTime, even online tools) and save it as `data/feynman_voice_sample.wav`. Make sure its mono, 22050 Hz WAV.

### Step 2: Install XTTS

```bash
pip install TTS soundfile
```

This downloads the TTS package from Coqui. First run will also download the XTTS v2 model (about 2 GB).

If pip install TTS fails on Mac, run `xcode-select --install` first.

### Step 3: Test it

```python
from TTS.api import TTS
import torch

# Pick device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load XTTS v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Generate speech cloning Feynman's voice from the 10 sec clip
tts.tts_to_file(
    text="Look, the thing is, nobody really understands quantum mechanics. And that is perfectly fine.",
    speaker_wav="data/feynman_voice_sample.wav",
    language="en",
    file_path="test_feynman_voice.wav"
)

print("Done. Listen to test_feynman_voice.wav")
```

Run it:

```bash
python -c "
from TTS.api import TTS
import torch
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
tts.tts_to_file(text='Look, the thing is, nobody really understands quantum mechanics.', speaker_wav='data/feynman_voice_sample.wav', language='en', file_path='test_feynman_voice.wav')
print('Done')
"
```

First run takes a few minutes to download the model. After that each generation takes 3 to 8 seconds.

### Step 4: Plug into the app

Replace `src/tts.py` with this:

```python
import os
import tempfile
import torch
from TTS.api import TTS

REFERENCE_AUDIO = os.path.join("data", "feynman_voice_sample.wav")

_tts = None

def _get_tts():
    global _tts
    if _tts is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts

def synthesize_speech(text, voice=None):
    tts = _get_tts()
    ref = voice or REFERENCE_AUDIO
    output_path = tempfile.mktemp(suffix=".wav", prefix="feynman_tts_")
    tts.tts_to_file(
        text=text,
        speaker_wav=ref,
        language="en",
        file_path=output_path
    )
    return output_path
```

Then restart the app with `python app.py`. Enable "Feynman Voice" toggle in the right panel.

The first message after restart will be slow (loading the 2GB model into memory). After that its 3 to 8 seconds per response.

## How it works

XTTS v2 extracts speaker characteristics (pitch, tone, cadence, accent) from your 10 second reference clip and applies them to any new text you generate. Its not perfect with just 10 seconds but it captures the general voice quality well enough.

For better results:
- Use a clip where Feynman speaks clearly and naturally (not shouting or whispering)
- Avoid clips with echo or background noise
- The BBC interviews ("Fun to Imagine", "The Pleasure of Finding Things Out") have the best audio quality

## If you want even better quality later

If you have more time, you can fine tune XTTS v2 on 30+ minutes of Feynman audio for a much more accurate voice clone. See the full fine tuning guide in the detailed `VOICE_CLONING_GUIDE.md`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pip install TTS` fails | Run `xcode-select --install` on Mac, then try again |
| Out of memory | Use `cpu` instead of `mps`/`cuda` (slower but works) |
| Voice sounds nothing like Feynman | Try a different 10 second clip with clearer audio |
| Model download stuck | Check internet, the model is about 2 GB |
| App crashes when voice is enabled | Make sure `data/feynman_voice_sample.wav` exists |
