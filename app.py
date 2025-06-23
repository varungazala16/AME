import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from TimedUpGo import analyze_tug_from_video
from HandPronation import count_flip_flops
from FistOpenClose import count_fist_openClose
from fingertap import count_taps
from RombergOutstretch import analyze_romberg_outstretch
from FootStomp import count_stomps_left,count_stomps_right
from FootTapLeft import analyze_left_foot_taps
from FootTapRight import analyze_right_foot_taps
from StandOnOneFootLeft import analyze_left_leg_stand
from StandOnOneFootRight import analyze_right_leg_stand
from AriseFromChair import analyze_sit_to_stand_RFC
from dual_attention import extract_audio_and_analyze_speech

app = Flask(__name__)

allowed_origins = [
    re.compile(r'http://localhost:\d+'),
    "https://automovementexam.netlify.app" 
]

CORS(app,
     origins=allowed_origins,
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])


def run_analysis_for_task(task_id, video_path):
    if task_id == 1:
        result = analyze_tug_from_video(video_path)  # Timed Up and Go

    elif task_id == 2:
        result = extract_audio_and_analyze_speech(video_path)  # Dual Attention

    elif task_id == 3:
        result = {"steps": 12}  # Tandem Gait

    elif task_id == 4:
        result = analyze_romberg_outstretch(video_path)  # Arms Outstretched Eyes Closed

    elif task_id == 5:
        result = analyze_right_leg_stand(video_path) # Stand On One Foot, Right

    elif task_id == 6:
        result = {"steps": 25}  # March to the Beat - Slow

    elif task_id == 7:
        result = analyze_right_foot_taps(video_path)  # Foot Tap, Right

    elif task_id == 8:
        result = count_taps(video_path)  # Finger Tap, Right

    elif task_id == 9:
        result = count_flip_flops(video_path, side='right')  # Hand Pronation, Right

    elif task_id == 10:
        result = count_fist_openClose(video_path)  # Fist Open and Close, Right

    elif task_id == 11:
        result = count_stomps_right(video_path)  # Foot Stomp, Right

    elif task_id == 12:
        result = {"steps": 28}  # March to the Beat - Fast

    elif task_id == 13:
        result = analyze_left_leg_stand(video_path)   # Stand On One Foot, Left

    elif task_id == 14:
        result = analyze_left_foot_taps(video_path) # Foot Tap, Left

    elif task_id == 15:
        result = count_taps(video_path)  # Finger Tap, Left

    elif task_id == 16:
        result = count_stomps_left(video_path)  # Foot Stomp, Left

    elif task_id == 17:
        result = count_flip_flops(video_path, side='left')  # Hand Pronation, Left

    elif task_id == 18:
        result = count_fist_openClose(video_path)  # Fist Open and Close, Left

    elif task_id == 19:
        result = analyze_sit_to_stand_RFC(video_path)  # Arise from chair

    else:
        result = {"error": "Unknown task ID"}

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