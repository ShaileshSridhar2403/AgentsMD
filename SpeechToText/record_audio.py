import pyaudio
import wave
import threading

# Audio recording parameters
FORMAT = pyaudio.paInt16  # 16-bit format
CHANNELS = 1  # Mono
RATE = 44100  # Sample rate (Hz)
CHUNK = 1024  # Buffer size
DEFAULT_FILENAME = "recorded_audio.wav"

def _record_audio(frames, stop_event):
    """Internal function to record audio until stop_event is set."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Error: {e}")
            break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

def record_audio_interactive(filename=DEFAULT_FILENAME):
    """Interactive function to start and stop recording with Enter key."""
    input("Press Enter to start recording...")  # Wait for Enter

    frames = []
    stop_event = threading.Event()
    thread = threading.Thread(target=_record_audio, args=(frames, stop_event))
    
    print("Recording... Press Enter to stop.")
    thread.start()

    input()  # Wait for Enter to stop
    stop_event.set()
    thread.join()

    # Save the recorded data to a WAV file
    with wave.open(filename, 'wb') as wf:
        audio = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio recorded and saved as '{filename}'.")

if __name__ == "__main__":
    record_audio_interactive("my_audio.wav")