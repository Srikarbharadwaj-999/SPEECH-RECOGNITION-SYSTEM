import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from pydub import AudioSegment

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path="converted.wav"):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    return wav_path

# Transcribe WAV using Wav2Vec2
def transcribe_wav2vec(audio_file): 
    if audio_file.lower().endswith(".mp3"):
        print(f"Converting {audio_file} to WAV...")
        audio_file = convert_mp3_to_wav(audio_file)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    print("\n🔊 Transcription:\n", transcription)
    return transcription

# Call function with the provided MP3 path
if __name__ == "__main__":
    transcribe_wav2vec("/content/ttsMP3.com_VoiceText_2025-5-15_19-7-50.mp3")
