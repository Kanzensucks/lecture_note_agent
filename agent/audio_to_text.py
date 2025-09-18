import whisper
import subprocess
import os

def transcribe_audio(file_path: str) -> str:
    model = whisper.load_model("tiny")  # or "base", "small"
    result = model.transcribe(file_path)
    return result["text"]

def extract_audio_from_video(video_path: str, output_folder: str = "data") -> str:
    audio_path = os.path.join(output_folder, "extracted_audio.wav")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path
