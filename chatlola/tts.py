import cloudinary.uploader
import cloudinary.api
import requests
from dotenv import load_dotenv
import os

load_dotenv()

cloudinary.config(
    cloud_name = "dv6bd4wgt",
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

def existing_tts(public_id):
    try:
        audio = cloudinary.api.resource(
            public_id,
            resource_type="video"
        )
        return audio["secure_url"]
    except cloudinary.exceptions.NotFound:
        return None

def uploadAudio(audio_bytes, public_id):

    audio_upload = cloudinary.uploader.upload(
            audio_bytes,
            resource_type="video",
            public_id=public_id)

    return audio_upload["secure_url"]

def getTTS(text):
    response = requests.post(
        "https://eidosspeech.xyz/api/v1/tts",
        headers={"X-API-Key": os.getenv("EIDOSPEECH_API_KEY")},
        json={
            "text": text,
            "voice": "fil-PH-BlessicaNeural",
            "format": "mp3",
            "bitrate": "128k",
            "rate": "-20%"
        }
    )

    if response.status_code == 200:
        print(f"Remaining: {response.headers.get('X-RateLimit-Remaining-Day')}")
        return response.content
    else:
        print(response.json())
        return None

