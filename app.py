"""PDF-to-Podcast: Drop a PDF, get a two-host podcast discussion as MP3."""

import json
import os
import re
import tempfile
from pathlib import Path

# Ensure Homebrew binaries (ffmpeg/ffprobe) are on PATH
for p in ["/opt/homebrew/bin", "/usr/local/bin"]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")

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
SILENCE_MS = 250

TTS_BACKEND = os.environ.get("TTS_BACKEND", "edge")  # "fish", "openrouter", or "edge" (free)

# Edge-TTS voices (free, no API key) — Indian English
EDGE_VOICE_A = "en-IN-PrabhatNeural"       # Arjun / Host A (male)
EDGE_VOICE_B = "en-IN-NeerjaExpressiveNeural"  # Priya / Host B (female, expressive)


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
SCRIPT_PROMPT = """You are a podcast script writer. Convert the following document into a deep, thorough, and engaging conversation between two Indian hosts:

- **Arjun** (Host A): Curious, witty, relatable. Asks sharp follow-up questions, uses everyday Indian analogies (cricket, chai, Bangalore traffic, startup culture, JEE prep, etc.), occasionally plays devil's advocate. Uses natural Indian English expressions like "yaar", "basically", "see the thing is", "no but seriously", "that's actually insane", "wait wait wait".
- **Priya** (Host B): The expert. Brilliant explainer, breaks down complex ideas with vivid examples. Uses phrases like "so here's the interesting part", "think of it this way", "exactly, and that's why", "the beauty of this is". Gives specific numbers and data from the paper.

Rules:
- 25-40 turns total — this should feel like a full podcast episode
- Each turn: 2-5 sentences. Conversational, like two IIT friends catching up over chai.
- Use natural filler words and reactions: "hmm", "right right", "oh wow", "okay so", "achha", "haan"
- Cover ALL major sections, findings, methodology, and conclusions
- Go deep: explain the WHY behind findings, not just the WHAT
- Arjun should ask "okay but real talk, what does this actually mean for..." type questions
- Priya should use concrete analogies Indians would get — compare scale to Indian cities, use cricket/Bollywood/startup metaphors
- Include genuine reactions — surprise, skepticism, excitement
- Start with Arjun setting up the topic casually
- Middle: walk through core ideas, with Arjun pushing for clarity and real-world relevance
- End with implications, limitations, and a compelling takeaway
- Sound like a real conversation, not a script. Interruptions and building on each other's points are good.
- Do NOT use Hindi words beyond the natural expressions listed above

Output ONLY valid JSON — an array of objects:
[{"speaker": "Arjun", "text": "..."}, {"speaker": "Priya", "text": "..."}, ...]

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
    ollama_limit = 24_000
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
                "options": {"num_predict": 8192, "temperature": 0.7},
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
    is_host_a = speaker in ("Alex", "Arjun")
    if TTS_BACKEND == "fish":
        voice_id = HOST_A_VOICE if is_host_a else HOST_B_VOICE
        return _tts_fish_audio(text, voice_id)
    elif TTS_BACKEND == "openrouter":
        voice_id = HOST_A_VOICE if is_host_a else HOST_B_VOICE
        return _tts_openrouter(text, voice_id)
    else:
        voice = EDGE_VOICE_A if is_host_a else EDGE_VOICE_B
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
    """Free TTS via edge-tts with SSML prosody for natural speech."""
    import asyncio
    import edge_tts

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()

    # Add natural pauses after punctuation and slight rate variation
    ssml_text = text
    ssml_text = ssml_text.replace("...", '<break time="400ms"/>')
    ssml_text = ssml_text.replace(". ", '. <break time="200ms"/>')
    ssml_text = ssml_text.replace("? ", '? <break time="300ms"/>')
    ssml_text = ssml_text.replace("! ", '! <break time="200ms"/>')
    # Pause after filler words for naturalness
    for filler in ["hmm,", "right right,", "okay so,", "achha,", "haan,"]:
        ssml_text = ssml_text.replace(filler, filler + ' <break time="250ms"/>')

    rate = "+5%" if voice == EDGE_VOICE_A else "+0%"
    pitch = "+0Hz" if voice == EDGE_VOICE_A else "+2Hz"

    ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-IN">
  <voice name="{voice}">
    <prosody rate="{rate}" pitch="{pitch}">
      {ssml_text}
    </prosody>
  </voice>
</speak>"""

    async def _generate():
        communicate = edge_tts.Communicate(ssml, voice)
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
