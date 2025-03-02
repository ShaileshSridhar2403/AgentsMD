import os
import sys
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response
from dotenv import load_dotenv
import werkzeug
import flask
import shutil
import threading
import re
import pdb
sys.path.append(os.path.abspath(".."))
from SpeechToText import RecordingManager
from SpeechToText import assembly_request

# Add parent directory to path so we can import the AI Triage System
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the AI Triage System
from ai_triage import ClinicalTriageSystem

# Add this at the top of your file
try:
    from werkzeug.urls import url_quote
except ImportError:
    # For newer Werkzeug versions
    from werkzeug.utils import url_quote

# Or if url_quote is completely removed, you can use urllib.parse
import urllib.parse

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Create output directories within the demo folder
os.makedirs("demo/quick_ref", exist_ok=True)
os.makedirs("demo/results", exist_ok=True)
os.makedirs("demo/discussions", exist_ok=True)

# Store the latest results for display
latest_results = {
    "case_id": None,
    "quick_ref_file": None,
    "detailed_output_file": None,
    "discussion_file": None
}

# Store progress updates
progress_updates = {
    "current_task": "Initializing...",
    "percentage": 0,
    "status": "pending",
    "message": ""
}

# Add this right after the imports
print(f"Flask version: {flask.__version__}")
print(f"Werkzeug version: {werkzeug.__version__}")

# Add this after creating the Flask app
recording_manager = RecordingManager()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/progress')
def progress():
    """Server-sent events for progress updates"""
    def generate():
        while progress_updates["status"] == "pending":
            data = json.dumps({
                "task": progress_updates["current_task"],
                "percentage": progress_updates["percentage"],
                "message": progress_updates["message"]
            })
            yield f"data: {data}\n\n"
            time.sleep(0.5)
        
        # Send final update
        data = json.dumps({
            "task": progress_updates["current_task"],
            "percentage": 100,
            "message": progress_updates["message"],
            "status": progress_updates["status"]
        })
        yield f"data: {data}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

def process_case_task(patient_case, api_key):
    """Background task to process the patient case"""
    try:
        # Reset progress
        progress_updates["current_task"] = "Initializing triage system..."
        progress_updates["percentage"] = 5
        progress_updates["status"] = "pending"
        progress_updates["message"] = "Setting up the AI agents..."
        
        # Define a progress callback function
        def update_progress(message, percentage=None):
            if percentage is not None:
                progress_updates["percentage"] = percentage
            progress_updates["message"] = message
        
        # Initialize the triage system
        triage_system = ClinicalTriageSystem(
            api_key=api_key,  # Use OpenAI API key
            llm_backend="o1-mini",  # Use OpenAI o1-mini
            verbose=True
        )
        
        # Monkey patch the AgentDiscussion.deliberate method to capture progress
        original_deliberate = triage_system.discussion.deliberate
        
        def patched_deliberate(conversation_text, case_id=None):
            # Update the current task
            progress_updates["current_task"] = "Agent Discussion"
            
            # Call the original method with our progress callback
            return original_deliberate(
                conversation_text=conversation_text,
                case_id=case_id,
                progress_callback=update_progress
            )
        
        # Replace the method
        triage_system.discussion.deliberate = patched_deliberate
        
        # Process the conversation
        results = triage_system.process_conversation(patient_case)
        
        # Update progress for file operations
        progress_updates["current_task"] = "Generating Output"
        progress_updates["percentage"] = 90
        progress_updates["message"] = "Creating output files..."
        
        case_id = results["case_id"]
        latest_results["case_id"] = case_id
        
        # Create demo directories
        os.makedirs("demo/quick_ref", exist_ok=True)
        os.makedirs("demo/results", exist_ok=True)
        os.makedirs("demo/discussions", exist_ok=True)
        
        # Copy quick reference files
        quick_ref_dir = "quick_ref"
        quick_ref_files = [f for f in os.listdir(quick_ref_dir) if f.startswith(case_id)]
        for file in quick_ref_files:
            src = os.path.join(quick_ref_dir, file)
            dst = os.path.join("demo/quick_ref", file)
            shutil.copy2(src, dst)
            latest_results["quick_ref_file"] = dst
        
        # Copy result files
        results_dir = "results"
        result_files = [f for f in os.listdir(results_dir) if f.startswith(case_id)]
        for file in result_files:
            src = os.path.join(results_dir, file)
            dst = os.path.join("demo/results", file)
            shutil.copy2(src, dst)
            if file.endswith(".txt"):
                latest_results["detailed_output_file"] = dst
        
        # Copy discussion files
        discussions_dir = "discussions"
        discussion_files = [f for f in os.listdir(discussions_dir) if f.startswith(case_id)]
        for file in discussion_files:
            src = os.path.join(discussions_dir, file)
            dst = os.path.join("demo/discussions", file)
            shutil.copy2(src, dst)
            latest_results["discussion_file"] = dst
        
        # Update progress
        progress_updates["current_task"] = "Complete"
        progress_updates["percentage"] = 100
        progress_updates["message"] = f"Triage assessment complete! ESI Level: {results['esi_level']}"
        progress_updates["status"] = "complete"
        
    except Exception as e:
        # Update progress with error
        progress_updates["current_task"] = "Error"
        progress_updates["percentage"] = 100
        progress_updates["message"] = str(e)
        progress_updates["status"] = "error"

@app.route('/process', methods=['POST'])
def process_case():
    """Process a patient case"""
    # Get the patient case text from the form
    print(request.form)
    # pdb.set_trace()
    patient_case = request.form.get('conversation_text', '')
    
    if not patient_case:
        return jsonify({"error": "No patient case provided"}), 400
    
    # Get API key from environment - use OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return jsonify({"error": "No OpenAI API key found in environment. Please set OPENAI_API_KEY in .env file."}), 400
    
    # Start processing in a background thread
    thread = threading.Thread(target=process_case_task, args=(patient_case, api_key))
    thread.daemon = True
    thread.start()
    
    # Return immediately with a success message
    return jsonify({"success": True, "message": "Processing started"})

@app.route('/check_status')
def check_status():
    """Check the status of the processing"""
    if progress_updates["status"] == "complete":
        # Read the quick reference file for immediate display
        quick_ref_content = ""
        esi_level = None
        if latest_results["quick_ref_file"] and os.path.exists(latest_results["quick_ref_file"]):
            with open(latest_results["quick_ref_file"], 'r') as f:
                quick_ref_content = f.read()
                # Extract ESI level from the content
                esi_match = re.search(r'ESI LEVEL: (\d)', quick_ref_content)
                if esi_match:
                    esi_level = esi_match.group(1)
        
        return jsonify({
            "status": "complete",
            "case_id": latest_results["case_id"],
            "quick_ref": quick_ref_content,
            "esi_level": esi_level,
            "has_detailed_output": latest_results["detailed_output_file"] is not None,
            "has_discussion": latest_results["discussion_file"] is not None
        })
    elif progress_updates["status"] == "error":
        return jsonify({
            "status": "error",
            "message": progress_updates["message"]
        })
    else:
        return jsonify({
            "status": "pending",
            "task": progress_updates["current_task"],
            "percentage": progress_updates["percentage"],
            "message": progress_updates["message"]
        })

@app.route('/view_detailed_output')
def view_detailed_output():
    """View the detailed output file"""
    if not latest_results["detailed_output_file"] or not os.path.exists(latest_results["detailed_output_file"]):
        return jsonify({"error": "No detailed output file available"}), 404
    
    with open(latest_results["detailed_output_file"], 'r') as f:
        content = f.read()
    
    return jsonify({"content": content})

@app.route('/view_discussion')
def view_discussion():
    """View the discussion file"""
    if not latest_results["discussion_file"] or not os.path.exists(latest_results["discussion_file"]):
        return jsonify({"error": "No discussion file available"}), 404
    
    with open(latest_results["discussion_file"], 'r') as f:
        content = f.read()
    
    return jsonify({"content": content})

@app.route('/download/<file_type>')
def download_file(file_type):
    """Download a file"""
    if file_type == 'quick_ref' and latest_results["quick_ref_file"]:
        return send_file(
            latest_results["quick_ref_file"], 
            as_attachment=True,
            download_name=os.path.basename(latest_results["quick_ref_file"])
        )
    elif file_type == 'detailed_output' and latest_results["detailed_output_file"]:
        return send_file(
            latest_results["detailed_output_file"], 
            as_attachment=True,
            download_name=os.path.basename(latest_results["detailed_output_file"])
        )
    elif file_type == 'discussion' and latest_results["discussion_file"]:
        return send_file(
            latest_results["discussion_file"], 
            as_attachment=True,
            download_name=os.path.basename(latest_results["discussion_file"])
        )
    else:
        return jsonify({"error": "File not found"}), 404

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
def transcribe_audio():
    # Create transcriptions directory if it doesn't exist
    os.makedirs('transcriptions', exist_ok=True)
    
    filename = f"recordings/audio.wav"
    transcription = assembly_request.transcribe_audio(filename, "transcriptions/transcription.txt")
    return jsonify({
        "status": "success",
        "transcription": transcription
    }), 200

if __name__ == '__main__':
    # Run the app
    app.run(debug=True) 