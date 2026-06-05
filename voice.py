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