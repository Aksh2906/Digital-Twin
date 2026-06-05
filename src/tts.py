import os
import tempfile
import asyncio
import edge_tts

VOICE = os.getenv("FEYNMAN_TTS_VOICE", "en-US-RogerNeural")


async def _synthesize_async(text, output_path):
    communicate = edge_tts.Communicate(text, VOICE, rate="+5%", pitch="-2Hz")
    await communicate.save(output_path)


def synthesize_speech(text, voice=None):
    if voice is None:
        voice = VOICE

    output_path = tempfile.mktemp(suffix=".mp3", prefix="feynman_tts_")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(_synthesize_async(text, output_path))).result()
        else:
            loop.run_until_complete(_synthesize_async(text, output_path))
    except RuntimeError:
        asyncio.run(_synthesize_async(text, output_path))

    return output_path



# import os
# import tempfile
# import torch
# from TTS.api import TTS

# REFERENCE_AUDIO = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "data", "feynman_voice_sample.wav"
# )

# _tts = None

# def _get_tts():
#     global _tts
#     if _tts is None:
#         device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
#         _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
#     return _tts

# def synthesize_speech(text, voice=None):
#     tts = _get_tts()
#     ref = voice or REFERENCE_AUDIO
#     output_path = tempfile.mktemp(suffix=".wav", prefix="feynman_tts_")
#     tts.tts_to_file(
#         text=text,
#         speaker_wav=ref,
#         language="en",
#         file_path=output_path
#     )
#     return output_path