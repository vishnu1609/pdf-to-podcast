"""PDF-to-Podcast: Drop a PDF, get a two-host podcast discussion as MP3."""

import json
import os
import re
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import gradio as gr
import httpx
from pydub import AudioSegment

# --- Config ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FISH_AUDIO_API_KEY = os.environ.get("FISH_AUDIO_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Fish Audio voice reference IDs (browse https://fish.audio for more)
HOST_A_VOICE = os.environ.get("HOST_A_VOICE", "")  # leave empty for default
HOST_B_VOICE = os.environ.get("HOST_B_VOICE", "")

SCRIPT_MODEL = "claude-sonnet-4-20250514"
MAX_PDF_PAGES = 50
SILENCE_MS = 400

TTS_BACKEND = os.environ.get("TTS_BACKEND", "edge")  # "fish", "openrouter", or "edge" (free)

# Edge-TTS voices (free, no API key)
EDGE_VOICE_A = "en-US-GuyNeural"
EDGE_VOICE_B = "en-US-JennyNeural"


# --- Step 1: PDF → Structured Text ---
def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = min(len(doc), MAX_PDF_PAGES)
    chunks = []
    for i in range(pages):
        page = doc[i]
        text = page.get_text("text")
        if text.strip():
            chunks.append(f"[Page {i+1}]\n{text.strip()}")
    doc.close()
    if not chunks:
        raise ValueError("No text extracted — scanned/image PDFs not yet supported.")
    return "\n\n".join(chunks)


# --- Step 2: Text → Conversation Script ---
SCRIPT_PROMPT = """You are a podcast script writer. Convert the following document into a natural, engaging conversation between two hosts:

- **Alex** (Host A): Curious, asks good questions, makes analogies to everyday things
- **Sam** (Host B): The expert, explains clearly, gives key numbers and insights

Rules:
- 8-15 turns total
- Each turn: 1-3 sentences. Natural speech, not reading aloud.
- Cover the main ideas, skip boilerplate
- Start with Alex introducing the topic in one excited sentence
- End with Sam giving the "one thing to remember" takeaway
- Sound like two friends talking, not a lecture

Output ONLY valid JSON — an array of objects:
[{"speaker": "Alex", "text": "..."}, {"speaker": "Sam", "text": "..."}, ...]

No markdown fences, no commentary."""


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")  # e.g. "llama3.1" or "qwen2.5"


def generate_script(document_text: str) -> list[dict]:
    text_limit = 80_000
    if len(document_text) > text_limit:
        document_text = document_text[:text_limit] + "\n\n[...truncated]"

    if ANTHROPIC_API_KEY:
        return _script_via_anthropic(document_text)
    elif OPENAI_API_KEY:
        return _script_via_openai(document_text)
    elif OLLAMA_MODEL:
        return _script_via_ollama(document_text)
    else:
        # Try Ollama auto-detect
        detected_model = None
        try:
            resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
            models = resp.json().get("models", [])
            if models:
                detected_model = models[0]["name"]
        except Exception as e:
            print(f"  Ollama auto-detect failed: {e}")
        if detected_model:
            return _script_via_ollama(document_text, detected_model)
        raise ValueError(
            "No LLM backend found. Set one of:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  OPENAI_API_KEY=sk-...\n"
            "  OLLAMA_MODEL=llama3.1  (with Ollama running)\n"
            "Or start Ollama: ollama serve && ollama pull llama3.1"
        )


def _script_via_anthropic(text: str) -> list[dict]:
    import anthropic

    kwargs = {}
    if ANTHROPIC_API_KEY:
        kwargs["api_key"] = ANTHROPIC_API_KEY
    client = anthropic.Anthropic(**kwargs)
    resp = client.messages.create(
        model=SCRIPT_MODEL,
        max_tokens=4096,
        system=SCRIPT_PROMPT,
        messages=[{"role": "user", "content": f"Here is the document:\n\n{text}"}],
    )
    return _parse_script(resp.content[0].text)


def _script_via_openrouter(text: str) -> list[dict]:
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "anthropic/claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": SCRIPT_PROMPT},
                {"role": "user", "content": f"Here is the document:\n\n{text}"},
            ],
            "max_tokens": 4096,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return _parse_script(resp.json()["choices"][0]["message"]["content"])


def _script_via_openai(text: str) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SCRIPT_PROMPT},
            {"role": "user", "content": f"Here is the document:\n\n{text}"},
        ],
        max_tokens=4096,
    )
    return _parse_script(resp.choices[0].message.content)


def _script_via_ollama(text: str, model: str | None = None) -> list[dict]:
    model = model or OLLAMA_MODEL
    # 8B models need shorter input to leave room for output
    ollama_limit = 12_000
    if len(text) > ollama_limit:
        text = text[:ollama_limit] + "\n\n[...truncated for brevity]"

    for attempt in range(3):
        resp = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SCRIPT_PROMPT},
                    {"role": "user", "content": f"Here is the document:\n\n{text}"},
                ],
                "stream": False,
                "options": {"num_predict": 4096, "temperature": 0.7},
            },
            timeout=300,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        try:
            return _parse_script(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Ollama attempt {attempt+1}/3 failed to produce valid JSON: {e}")
            print(f"  Raw output preview: {content[:200]}")
            if attempt == 2:
                raise ValueError(
                    f"Ollama ({model}) failed to produce valid JSON after 3 attempts. "
                    f"Last output: {content[:300]}"
                )


def _parse_script(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    # Extract JSON array even if surrounded by commentary
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    script = json.loads(raw)
    if not isinstance(script, list) or not all(
        "speaker" in t and "text" in t for t in script
    ):
        raise ValueError("Script JSON must be [{speaker, text}, ...]")
    return script


# --- Step 3: Script → Audio ---
def synthesize_speech(text: str, speaker: str) -> bytes:
    if TTS_BACKEND == "fish":
        voice_id = HOST_A_VOICE if speaker == "Alex" else HOST_B_VOICE
        return _tts_fish_audio(text, voice_id)
    elif TTS_BACKEND == "openrouter":
        voice_id = HOST_A_VOICE if speaker == "Alex" else HOST_B_VOICE
        return _tts_openrouter(text, voice_id)
    else:
        voice = EDGE_VOICE_A if speaker == "Alex" else EDGE_VOICE_B
        return _tts_edge(text, voice)


def _tts_fish_audio(text: str, voice_id: str) -> bytes:
    body = {"text": text, "format": "mp3"}
    if voice_id:
        body["reference_id"] = voice_id
    resp = httpx.post(
        "https://api.fish.audio/v1/tts",
        headers={
            "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def _tts_openrouter(text: str, voice_id: str) -> bytes:
    resp = httpx.post(
        "https://openrouter.ai/api/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "fish-audio/drama-3",
            "input": text,
            "voice": voice_id or "default",
            "response_format": "mp3",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def _tts_edge(text: str, voice: str) -> bytes:
    """Free TTS via edge-tts (Microsoft Edge, no API key needed)."""
    import asyncio
    import edge_tts

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()

    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp.name)

    asyncio.run(_generate())
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data


# --- Step 4: Stitch audio segments ---
def stitch_audio(segments: list[bytes]) -> str:
    silence = AudioSegment.silent(duration=SILENCE_MS)
    podcast = AudioSegment.empty()

    for i, audio_bytes in enumerate(segments):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        segment = AudioSegment.from_mp3(tmp.name)
        os.unlink(tmp.name)
        if i > 0:
            podcast += silence
        podcast += segment

    out_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    podcast.export(out_path, format="mp3", bitrate="192k")
    return out_path


# --- Pipeline ---
def pdf_to_podcast(pdf_file, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")

    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    progress(0.05, "Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    page_count = text.count("[Page ")

    progress(0.2, "Writing conversation script...")
    script = generate_script(text)
    turn_count = len(script)

    audio_segments = []
    for i, turn in enumerate(script):
        frac = 0.3 + (0.6 * i / turn_count)
        progress(frac, f"Voice {i+1}/{turn_count}: {turn['speaker']}...")
        audio = synthesize_speech(turn["text"], turn["speaker"])
        audio_segments.append(audio)

    progress(0.95, "Stitching audio...")
    output_path = stitch_audio(audio_segments)

    transcript = "\n\n".join(f"**{t['speaker']}:** {t['text']}" for t in script)
    info = f"📄 {page_count} pages → 🎙️ {turn_count} turns | TTS: {TTS_BACKEND}"
    return output_path, transcript, info


# --- Gradio UI ---
def build_ui():
    with gr.Blocks(title="PDF to Podcast") as app:
        gr.Markdown("# 🎙️ PDF to Podcast\nDrop a PDF → get a two-host podcast discussion")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                generate_btn = gr.Button("Generate Podcast", variant="primary", size="lg")
                info_box = gr.Textbox(label="Info", interactive=False)

            with gr.Column(scale=2):
                audio_output = gr.Audio(label="Podcast", type="filepath")
                transcript_output = gr.Markdown(label="Transcript")

        generate_btn.click(
            fn=pdf_to_podcast,
            inputs=[pdf_input],
            outputs=[audio_output, transcript_output, info_box],
        )

    return app


if __name__ == "__main__":
    # Startup check
    backend = "none"
    if ANTHROPIC_API_KEY:
        backend = "Anthropic"
    elif OPENAI_API_KEY:
        backend = "OpenAI"
    elif OLLAMA_MODEL:
        backend = f"Ollama ({OLLAMA_MODEL})"
    else:
        try:
            r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
            models = r.json().get("models", [])
            if models:
                backend = f"Ollama auto ({models[0]['name']})"
        except Exception as e:
            print(f"  Ollama check failed: {e}")

    print(f"\n  LLM backend: {backend}")
    print(f"  TTS backend: {TTS_BACKEND}")
    if backend == "none":
        print("\n  ⚠️  No LLM backend found! Set one of:")
        print("     export ANTHROPIC_API_KEY=sk-ant-...")
        print("     export OPENAI_API_KEY=sk-...")
        print("     ollama serve && ollama pull llama3.1")
    print()

    app = build_ui()
    app.launch(share=False)
