from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from TimedUpGo import analyze_sit_to_stand
from fingertap import count_taps
from HandPronation import count_flip_flops
from FistOpenClose import count_fist_openClose
import os


# Initialize Supabase client
url = "https://nflthqflazkgwownewrc.supabase.co"
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

app = Flask(__name__)
CORS(app)

def run_analysis_for_task(task_id, video_path):
    if task_id == 1:
        result = analyze_sit_to_stand(video_path)
    elif task_id == 2:
        result = {"correct_count": 7}
    elif task_id == 3:
        result = {"steps": 12}
    elif task_id == 4:
        result = {"duration": 15}
    elif task_id == 5:
        result = {"duration": 8}
    elif task_id == 6:
        result = {"steps": 25}
    elif task_id == 7:
        result = mainFunction(video_path)
    elif task_id == 8:
        result = count_taps(video_path)
    elif task_id == 9:
        result = count_flip_flops(video_path, "right")
    elif task_id == 10:
        result = count_fist_openClose(video_path)
    else:
        result = ["44", "none"]
    return result

def get_latest_task_result_for_recording(recording_id):
    resp = supabase.table("task_results") \
        .select("*") \
        .eq("recording_id", recording_id) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()
    return resp.data[0] if resp.data else None

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    Expects JSON:
    {
        "recording_id": "...",
        "task_id": 1,
        "video_url": "..."
    }
    """
    data = request.get_json()
    recording_id = data.get("recording_id")
    task_id = data.get("task_id")
    video_url = data.get("video_url")
    
    if not (recording_id and task_id and video_url):
        return jsonify({"error": "Missing required fields"}), 400
    
    # 1. Run analysis
    metrics = run_analysis_for_task(task_id, video_url)
    
    # 2. Save result in task_results (upsert)
    supabase.table("task_results").upsert({
        "recording_id": recording_id,
        "metrics": metrics,
        "analysis_status": "complete"
    }).execute()
    
    # 3. Get latest result for this recording
    latest_result = get_latest_task_result_for_recording(recording_id)
    
    return jsonify({
        "recording_id": recording_id,
        "task_id": task_id,
        "latest_result": latest_result
    })



@app.route('/', methods=['GET'])
def analyze_video():
    print("Received test request")
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
    

