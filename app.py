import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from HandPronation import flip_flops
from FistOpenClose import count_fist_openClose
from fingertap import count_taps
from RombergOutstretch import analyze_romberg_outstretch
from FootStomp import count_stomps

app = Flask(__name__)

allowed_origins = [
    re.compile(r'http://localhost:\d+'),
    
    # Your production frontend URL (trailing slash removed)
    "https://automovementexam.netlify.app" 
]

CORS(app,
     origins=allowed_origins,
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])


def run_analysis_for_task(task_id, video_path):
    # This function is correct, no changes needed
    if task_id == 1:
        result = {"steps": 25}
    elif task_id == 2:
        result = {"correct_count": 7}
    elif task_id == 3:
        result = {"steps": 12}
    elif task_id == 4:
        result = analyze_romberg_outstretch(video_path)
    elif task_id == 5:
        result = {"duration": 8}
    elif task_id == 6:
        result = {"steps": 25}
    elif task_id == 7:
        result = {"steps": 25}
    elif task_id == 8:
        result = count_taps(video_path)
    elif task_id == 9:
        result = flip_flops(video_path)
    elif task_id == 10:
        result = count_fist_openClose(video_path)
    elif task_id == 11:
        result = count_stomps(video_path)
    else:
        result = {"error": "unknown task"}
    return result

@app.route('/analyze', methods=['POST']) # Only 'POST' is needed here
def analyze_single_recording():
    try:
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        task_id = data.get('task_id')
        video_path = data.get('recording_url')
        recording_id = data.get('recording_id')

        if not all([task_id, video_path, recording_id]):
            return jsonify({"error": "Missing required fields"}), 400

        result = run_analysis_for_task(task_id, video_path)

        return jsonify({
            "recording_id": recording_id,
            "task_id": task_id,
            "metrics": result
        })
    
    except Exception as e:
        import traceback
        print("🔥 Exception occurred:", traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)