import whisper
import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
from elevenlabs import ElevenLabs, play
import os

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

whisper_model = whisper.load_model("base")
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
vad = webrtcvad.Vad(2)  # aggressiveness 0-3

# Find Feynman voice ID on ElevenLabs or use a similar voice
VOICE_ID = "your_voice_id_here"

def record_until_silence(max_silence_frames=30):
    """Records audio, stops after silence detected."""
    audio_buffer = []
    silence_count = 0
    speaking = False
    q = queue.Queue()

    def callback(indata, frames, time, status):
        q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype='int16', blocksize=FRAME_SIZE,
                        callback=callback):
        print("Listening...")
        while True:
            frame = q.get()
            frame_bytes = frame.tobytes()

            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)

            if is_speech:
                speaking = True
                silence_count = 0
                audio_buffer.append(frame)
            elif speaking:
                silence_count += 1
                audio_buffer.append(frame)
                if silence_count > max_silence_frames:
                    break

    return np.concatenate(audio_buffer, axis=0)

def transcribe(audio_array):
    """Whisper transcription."""
    audio_float = audio_array.flatten().astype(np.float32) / 32768.0
    result = whisper_model.transcribe(audio_float, language="en")
    return result["text"].strip()

def speak(text, interrupt_event):
    """Convert text to Feynman voice, stoppable on interrupt."""
    audio = eleven_client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_monolingual_v1"
    )
    # Play in chunks, check interrupt between chunks
    for chunk in audio:
        if interrupt_event.is_set():
            print("Interrupted!")
            break
        play(chunk)

def voice_loop(chat_fn):
    """Main hands-free loop."""
    print("Feynman Voice Mode. Speak to begin.\n")

    while True:
        # Listen for user input
        audio = record_until_silence()
        text = transcribe(audio)

        if not text:
            continue

        print(f"You: {text}")

        # Get Feynman response
        response = chat_fn(text)
        print(f"Feynman: {response}")

        # Speak response, with interrupt detection running in parallel
        interrupt_event = threading.Event()

        def listen_for_interrupt():
            """If user speaks while Feynman is talking, interrupt."""
            interrupt_audio = record_until_silence(max_silence_frames=5)
            interrupt_text = transcribe(interrupt_audio)
            if interrupt_text:
                interrupt_event.set()
                # Put this back as next user input
                print(f"\n[Interrupted] You: {interrupt_text}")
                return interrupt_text
            return None

        speak_thread = threading.Thread(target=speak, args=(response, interrupt_event))
        interrupt_thread = threading.Thread(target=listen_for_interrupt)

        speak_thread.start()
        interrupt_thread.start()
        speak_thread.join()
        interrupt_thread.join()