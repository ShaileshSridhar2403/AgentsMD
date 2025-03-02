from .record_audio import record_audio_interactive
from .assembly_request import transcribe_audio

def record_and_transcribe_audio():
    record_audio_interactive("speech.wav")
    print("Audio recorded successfully")
    transcribe_audio("speech.wav")
    print("Audio transcribed successfully")

if __name__ == "__main__":
    record_and_transcribe_audio()
