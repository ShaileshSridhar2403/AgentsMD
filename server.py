from flask import Flask, jsonify
from SpeechToText import RecordingManager
from SpeechToText import assembly_request
import os

app = Flask(__name__)
recording_manager = RecordingManager()

@app.route('/recorder/start_recording', methods=['POST'])
def start_recording():
    success = recording_manager.start_recording()
    if success:
        return jsonify({"status": "success", "message": "Recording started"}), 200
    else:
        return jsonify({"status": "error", "message": "Recording already in progress"}), 400

@app.route('/recorder/stop_recording', methods=['POST'])
def stop_recording():
    # Create recordings directory if it doesn't exist
    os.makedirs('recordings', exist_ok=True)
    
    # Generate unique filename using timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/audio.wav"
    
    success = recording_manager.stop_recording(filename)
    if success:
        return jsonify({
            "status": "success", 
            "message": "Recording stopped and saved",
            "filename": filename
        }), 200
    else:
        return jsonify({
            "status": "error", 
            "message": "No recording in progress"
        }), 400

@app.route('/recorder/recording_status', methods=['GET'])
def recording_status():
    is_recording = recording_manager.is_currently_recording()
    return jsonify({
        "status": "success",
        "is_recording": is_recording
    }), 200

@app.route('/transcriber/transcribe', methods=['POST'])
def transcribe():
    filename = f"recordings/audio.wav"
    transcription = assembly_request.transcribe_audio(filename, "transcriptions/transcription.txt")
    return jsonify({
        "status": "success",
        "transcription": transcription
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
