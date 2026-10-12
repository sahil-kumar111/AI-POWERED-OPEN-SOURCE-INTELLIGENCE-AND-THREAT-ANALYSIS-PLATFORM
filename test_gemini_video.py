from google import genai
from pathlib import Path

video_path = Path("input/YOUTUBE_YouTubeCCTV196_gun_2.mp4")

client = genai.Client(
    api_key=__import__("os").environ["GEMINI_API_KEY"]
)

video = client.files.upload(file=str(video_path))

import time

while video.state.name == "PROCESSING":
    time.sleep(2)
    video = client.files.get(name=video.name)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=[
        video,
        "Describe what happens visually in this CCTV video. "
        "Focus on people, objects, and potentially dangerous actions. "
        "Do not infer details that are not visible."
    ]
)

print(response.text)