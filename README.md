# SPEECH-RECOGNITION-SYSTEM

*COMPANY: CODITECH IT SOLUTIONS

*NAME: SRIKAR BHARADWAJ

*INTERN ID: CT08DM833

*DOMAIN: ARTIFICIAL INTELLIGENCE

*DURATION: 8 WEEKS

*MENTOR: NEELA SANTOSH

*DESCRIPTION: This program is designed to convert and transcribe speech from an MP3 audio file into text using a pre-trained model called Wav2Vec2. The program works in two main steps:

Convert MP3 to WAV:

MP3 files are not directly usable by the transcription model, so we first convert the MP3 audio into a WAV format.

The program uses a library called pydub to convert the MP3 to WAV. It also makes sure the audio is at the correct quality (16,000 samples per second) and in mono format (one channel of sound).

Transcribe the WAV file to text:

Once the audio is in WAV format, the program uses a deep learning model (Wav2Vec2) to listen to the audio and convert it into text.

This model has been trained on lots of audio data, so it can recognize words and transcribe them accurately.

How it works:
Step 1: The program checks if your input file is an MP3 file.

Step 2: If it is, it converts the MP3 file to WAV.

Step 3: Then, it loads the WAV file, processes it, and passes it to the Wav2Vec2 model to transcribe the spoken words into text.

Step 4: The transcribed text is printed on the screen, so you can read what was said in the audio.

Key Libraries Used:
pydub: Helps to convert MP3 files to WAV.

torchaudio: Used to read the audio and adjust its quality for transcription.

transformers: This is where the Wav2Vec2 model comes from. It’s a pre-trained model for speech recognition.

torch: This is used for deep learning and running the Wav2Vec2 model.

Example Use:
If you have an MP3 file with a person speaking (e.g., a recording of a paragraph), you can run the program, and it will print the text version of what was said in the audio
