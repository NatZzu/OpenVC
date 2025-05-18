#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# OpenVc – an ultra-lightweight, cloud-powered voice chat template for LLMs.
#
# Copyright 2025 Ascendant Softworks (Private) Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------


import logging
import os
import re
import requests
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Your Groq API key: set this in your environment or replace with a config loader.
groq_api_key = os.getenv("GROQ_API_KEY", "YOUR_API_KEY_HERE.")

# System-level prompt for the LLM: each query will prepend this. This will affect your LLM's personality, change this for your use-case. Make sure to keep/edit lines 2 & 3 according to your requirements for better accuracy.
sys_msg = (
    'You are OpenVc, an open-source, cloud-powered voice chat system template. '
    'You receive spoken prompts via STT (Whisper) and reply via TTS (PlayAI). '
    'Keep messages short and simple for voice chat.'
)

# Configure root logger for INFO-level output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ──────────────────────────────────────────────────────────────────────────────
# 1) SPEECH-TO-TEXT (STT) – Groq Whisper
# ──────────────────────────────────────────────────────────────────────────────

def process_voice_input(timeout: float = 5.0) -> str:
    """
    Listens for user speech via the microphone.
    The recognizer is set with fixed energy and pause thresholds so that:
      1, As soon as the mic volume exceeds the threshold, the recording begins.
      2, When the volume drops to a specific threshold, it stops listening.
    The audio is then sent to Groq's STT endpoint and the transcribed text is returned. Replace the STT endpoint with a local STT model if necessary.
    """
    recognizer = sr.Recognizer()
    # Fix energy threshold & disable dynamic adjustment for speedy start. This is necessary for faster latency, you can play around with these values according to your results.
    recognizer.energy_threshold = 400 # edit according to the background noise detected by your microphone.
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = 0.5  # seconds of silence to end

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        logging.info("🔊 Waiting for voice…")
        try:
            audio = recognizer.listen(source, timeout=timeout)
        except Exception as e:
            logging.error(f"STT capture error: {e}")
            return ""

    # Write to disk for upload.
    audio_path = "prompt.wav"
    with open(audio_path, "wb") as f:
        f.write(audio.get_wav_data())

    # Call Groq Whisper.
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {groq_api_key}"},
            files={
                "file": (audio_path, open(audio_path, "rb")),
                "model": (None, "distil-whisper-large-v3-en"), # change the model name to Whisper Large v3 for multi-lingual support. Refer to the Groq documentation for more info.
                "response_format": (None, "text")
            }
        )
        r.raise_for_status()
        text = r.text.strip()
        logging.info(f"👤 USER: {text}")
    except Exception as e:
        logging.error(f"STT API error: {e}")
        text = ""
    finally:
        os.remove(audio_path)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# 2) TEXT-TO-SPEECH (TTS) – Groq PlayAI-TTS
# ──────────────────────────────────────────────────────────────────────────────

def text_to_speech(text: str):
    """
    Send `text` to Groq PlayAI TTS and play back the returned WAV. Replace this logic with a local TTS model if needed.
    """
    audio_path = Path("speech.wav")
    payload = {
        "model": "playai-tts",
        "voice": "Atlas-PlayAI", # You can change the voice here. Refer to the Groq documentation for different voices.
        "input": text,
        "response_format": "wav"
    }
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    r = requests.post("https://api.groq.com/openai/v1/audio/speech", json=payload, headers=headers)
    if r.status_code == 200:
        audio_path.write_bytes(r.content)
        logging.info(f"🔈 Playing: {text}")
        play(AudioSegment.from_wav(audio_path))
        audio_path.unlink()
    else:
        logging.error(f"TTS API {r.status_code}: {r.text}")


# ──────────────────────────────────────────────────────────────────────────────
# 3) LLM QUERY – Groq Chat Completion
# ──────────────────────────────────────────────────────────────────────────────

def query_llm(prompt: str, max_tokens: int = 64) -> str:
    """
    Send `prompt` to Groq’s chat API, including the system message
    (with optional personalization), and return the assistant’s reply.
    """
    # Personalize if we have a name.
    prompt_sys = sys_msg
    if recognized_user_name:
        prompt_sys += f" The user's name is {recognized_user_name}."

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      json=payload,
                      headers={"Authorization": f"Bearer {groq_api_key}",
                               "Content-Type": "application/json"})
    r.raise_for_status()
    reply = r.json()['choices'][0]['message']['content'].strip()
    logging.info(f"🤖 LLM: {reply}")
    return reply


# ──────────────────────────────────────────────────────────────────────────────
# 4) MAIN: Name Input + Wake-Word Loop
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 4.1 – personalization: get name at startup.
    recognized_user_name = input("➡️  Please enter your name: ").strip()
    if recognized_user_name:
        logging.info(f"👋 Welcome, {recognized_user_name}!")
    else:
        logging.info("👋 Proceeding anonymously.")

    # 4.2 – wake-word patterns: only full-word “stop” or “start”.
    stop_pattern = re.compile(r'^\W*stop\W*$', re.IGNORECASE)
    start_pattern = re.compile(r'^\W*start\W*$', re.IGNORECASE)

    listening_enabled = True

    logging.info("✅ OpenVc is now running. Say ‘stop’ to pause, ‘start’ to resume.")
    while True:
        raw = process_voice_input()
        if not raw:
            continue

        # 4.3 – handle wake-words.
        if listening_enabled and stop_pattern.match(raw):
            text_to_speech("Voice chat stopped. Say ‘start’ to resume.")
            listening_enabled = False
            continue

        if not listening_enabled and start_pattern.match(raw):
            text_to_speech("Voice chat resumed.")
            listening_enabled = True
            continue

        # 4.4 – if active, send entire user utterance to LLM.
        if listening_enabled:
            reply = query_llm(raw)
            text_to_speech(reply)
        # else: paused and heard non-“start”—ignore silently.
