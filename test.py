import requests

url = "http://127.0.0.1:10000/analyze"
payload = {
    "recording_id": "812a2cfb-f68c-475f-89a0-76fa442313ac",
    "task_id": 10,
    "recording_url": "https://nflthqflazkgwownewrc.supabase.co/storage/v1/object/public/recordings//2025-06-17T02:24:57.286Z-task-10.webm-1750127097287"
}
response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Response:", response.json())
