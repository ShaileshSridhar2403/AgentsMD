import assemblyai as aai
import os
from dotenv import load_dotenv

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
# You can use a local filepath:
# audio_file = "./example.mp3"
# Or use a publicly-accessible URL:

def transcribe_audio(audio_file):
    config = aai.TranscriptionConfig(
    speaker_labels=True,
    speakers_expected=2
    )
    transcript = aai.Transcriber().transcribe(audio_file, config)
    dialogues = []
    for utterance in transcript.utterances:
        dialogues.append(f"Speaker {utterance.speaker}: {utterance.text}")
        print(f"Speaker {utterance.speaker}: {utterance.text}")
    with open("transcription.txt", "w") as f:
        f.write("\n".join(dialogues))
    return dialogues