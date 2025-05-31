import requests

data = {
    "recording_id": "38644931-8614-48dd-aaa9-33643df1aa86",
    "task_id": 9,
    "video_url": "https://nflthqflazkgwownewrc.supabase.co/storage/v1/object/public/recordings/1ebdbcdc-bf79-4a76-95e2-13f9fb0e7507/2025-05-31T19_09_08.487Z-task-9.webm"
}

resp = requests.post("http://localhost:5001/analyze", json=data)
print(resp.status_code)
print(resp.json())
