from flask import Flask, request, jsonify
from flask_cors import CORS
from TimedUpGo import analyze_sit_to_stand
from fingertap import count_taps
from HandPronation import count_flip_flops
from FistOpenClose import count_fist_openClose

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
        result = analyze_sit_to_stand(video_path)
    elif task_id == 8:
        result = count_taps(video_path)
    elif task_id == 9:
        result = count_flip_flops(video_path, "right")
    elif task_id == 10:
        result = count_fist_openClose(video_path)
    else:
        result = ["44", "none"]
    return result

@app.route('/analyze', methods=['POST'])
def analyze_recordings():
    """
    Expects JSON:
    {
      "recordings": [
        {
          "id": "recording_id",
          "task_id": 1,
          "recording_url": "https://..."
        },
        ...
      ]
    }
    """
    data = request.get_json()
    recordings = data.get('recordings', [])
    analysis_results = []

    for rec in recordings:
        task_id = rec.get('task_id')
        video_path = rec.get('recording_url')
        recording_id = rec.get('id')
        result = run_analysis_for_task(task_id, video_path)
        analysis_results.append({
            "recording_id": recording_id,
            "task_id": task_id,
            "metrics": result
        })

    return jsonify({"results": analysis_results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)