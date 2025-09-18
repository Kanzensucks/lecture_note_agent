from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
import logging
from typing import Optional

from agent.audio_to_text import transcribe_audio, extract_audio_from_video
from agent.pdf_reader import extract_text_from_pdf
from agent.summarize_agent import load_prompt, summarize, save_to_word

# ---------- Logging ----------
logger = logging.getLogger("lecture_note_runner")
if not logger.handlers:
    h = logging.StreamHandler()
    logger.addHandler(h)
logger.setLevel(logging.INFO)


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input_here"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS = Path.home() / "Downloads"


def get_latest_file(folder: Path, exts: tuple[str, ...]) -> Optional[Path]:
    files = [folder / f for f in os.listdir(folder) if f.endswith(exts)]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def main() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    pdf = get_latest_file(INPUT_DIR, (".pdf",))
    video = get_latest_file(INPUT_DIR, (".mp4", ".mov", ".mkv", ".avi"))
    audio = get_latest_file(INPUT_DIR, (".wav", ".mp3", ".m4a", ".flac"))

    if not any([pdf, video, audio]):
        logger.error("No input found. Drop a PDF, video, or audio file into: %s", INPUT_DIR)
        return

    transcript_text: Optional[str] = None
    src_display = ""
    src_path: Optional[Path] = None

    try:
        if pdf:
            src_path = pdf
            src_display = f"PDF: {pdf.name}"
            logger.info("Extracting text from %s", src_display)
            transcript_text = extract_text_from_pdf(str(pdf))
        elif video:
            src_path = video
            src_display = f"Video: {video.name}"
            logger.info("Extracting audio from %s", src_display)
            audio_path = extract_audio_from_video(str(video), output_folder=str(DATA_DIR))
            logger.info("Transcribing audio...")
            transcript_text = transcribe_audio(audio_path)
        elif audio:
            src_path = audio
            src_display = f"Audio: {audio.name}"
            logger.info("Transcribing %s", src_display)
            transcript_text = transcribe_audio(str(audio))

        assert transcript_text and transcript_text.strip(), "Empty transcript/text extracted."

        # Load prompts
        summary_prompt = load_prompt(str(PROMPTS_DIR / "summarise.txt"))
        notes_prompt = load_prompt(str(PROMPTS_DIR / "lecture_notes.txt"))

        # Run LLM via Ollama
        logger.info("Generating summary...")
        summary = summarize(transcript_text, summary_prompt)

        logger.info("Generating lecture notes...")
        notes = summarize(transcript_text, notes_prompt)

        # Save to Downloads
        base_name = sanitize_filename(src_path.stem if src_path else "output")
        summary_doc = DOWNLOADS / f"{base_name}_summary.docx"
        notes_doc = DOWNLOADS / f"{base_name}_lecture_notes.docx"
        save_to_word(summary, str(summary_doc))
        save_to_word(notes, str(notes_doc))

        print(f"Saved to:\n- {summary_doc}\n- {notes_doc}")

    finally:
        # Clean up input file to keep the folder tidy
        if src_path and src_path.exists():
            try:
                src_path.unlink()
                logger.info("Deleted input: %s", src_path)
            except Exception as e:
                logger.warning("Could not delete input file: %s", e)


if __name__ == "__main__":
    main()
