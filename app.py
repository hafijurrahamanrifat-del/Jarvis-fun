# JARVIS 5.0 ULTRA — free, no-key-first, agentic AI assistant for Streamlit
# Abilities: multi-mode brain (fast / reasoning / code), Deep Research with citations,
# web + page reading, YouTube, vision, files & tables, UNLIMITED image generation
# (auto-retry), AI VIDEO STUDIO (storyboard -> generated scenes -> narrated MP4),
# Bengali PDF export, voice I/O. No API key required to start.

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import ipaddress
import json
import os
import re
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

# --- Safe optional imports ---
try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None
try:
    from youtube_transcript_unblocked import YouTubeTranscriptApi as UnblockedYT
except Exception:
    UnblockedYT = None
try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None
try:
    import edge_tts
except Exception:
    edge_tts = None
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    from docx import Document
except Exception:
    Document = None
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from fpdf import FPDF
except Exception:
    FPDF = None

APP_VERSION = "5.0.0 Ultra"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "JARVIS-Ultimate/5.0"})
TIMEOUT = 90
MAX_CONTEXT_CHARS = 24000
MAX_PAGE_BYTES = 1_500_000
IMAGE_RETRY_WAITS = (3, 8, 20, 40)   # adaptive backoff through the free-tier quota window
VIDEO_W, VIDEO_H = 1280, 720

# --- Provider defaults (checked weekly by CI — see .github/workflows/model-drift.yml) ---
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"          # llama-3.3-70b-versatile was retired 2026-08-16
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"        # Google alias that auto-tracks the newest Flash
DEFAULT_OPENROUTER_MODELS = [                       # verified live in the public catalog
    "google/gemma-4-31b-it:free",                   # vision-capable
    "inclusionai/ling-3.2-flash:free",              # vision-capable
]
DEFAULT_POLLINATIONS_MODEL = "openai-fast"          # free, NO key — GPT-OSS 20B (reasoning)

BENGALI_FONT_URLS = [
    "https://github.com/google/fonts/raw/main/ofl/notosansbengali/NotoSansBengali%5Bwdth,wght%5D.ttf",
    "https://github.com/notofonts/bengali/releases/download/NotoSansBengali-v2.003/NotoSansBengali-v2.003.zip",
]

SYSTEM_PROMPT = """You are JARVIS 5.0, an elite personal AI agent running inside a Streamlit app with real tools.
Tool reports you may receive in CONTEXT: WEB RESULTS (search snippets with [n] numbers), SOURCE [n] (full page text),
YOUTUBE TRANSCRIPT, FILE (...), VIDEO STORYBOARD, and notes about images. Behavior rules:
1. Mirror the user's language (Bangla, English, or mixed). Be direct, structured, concise: short headings, bullets, fenced code blocks for code.
2. Ground answers in the provided CONTEXT when present and cite sources as [1], [2] matching the numbers. If the CONTEXT is insufficient, say honestly what you could not verify.
3. Where-to-watch questions: recommend official/legal platforms only (Netflix, Hoichoi, Chorki, SonyLIV, Crunchyroll, YouTube Movies). Never link piracy sites and never instruct ToS-violating automation.
4. Uploaded photos: if asked to edit while keeping the face identical, honestly explain that free AI generators create a new lookalike image, not the same face, and suggest professional tools. Otherwise analyze the attached image when asked.
5. CV/PDF requests: output clean, professional plain text (headings and bullets only) — it is auto-converted into a downloadable PDF.
6. Be honest about what you actually did this turn (searched the web, read pages, generated an image or video, or could not fetch something). Never fabricate links, citations, or tool results."""

MODE_PROMPTS = {
    "⚡ Fast": "",
    "🧠 Reasoning": "Reasoning mode: think through the problem step by step internally, then present a brief 'Reasoning' summary (2-4 bullets) followed by the final answer.",
    "💻 Code Expert": "Code expert mode: prefer complete, runnable code with imports and error handling, then a short explanation. State the language and any setup commands.",
}

# --- Intent detection (word-boundary regexes, no accidental substring triggers) ---
SEARCH_RE = re.compile(
    r"\b(?:search|google|find|look\s?up|latest|news|today|price|weather|who\s+is|what\s+is|"
    r"when|where|why|how\s+(?:much|many|to|does|do)|compare|best|top\s?\d+|review|release|score)\b"
    r"|খোঁজ|সার্চ|খবর|কোথায়|কিভাবে|কীভাবে|কখন|কেন|সর্বশেষ|আজকের",
    re.IGNORECASE,
)
MEDIA_RE = re.compile(
    r"\b(?:movie|film|anime|series|watch|stream|episode)\b|সিনেমা|অ্যানিমে|মুভি|দেখতে\s?চাই|দেখব",
    re.IGNORECASE,
)
GEN_IMG_RE = re.compile(
    r"\b(?:draw|generate|create|make|design)\b[^.?!]{0,50}\b(?:image|picture|photo|logo|poster|art|wallpaper|thumbnail|ছবি|স্কেচ)\b"
    r"|ছবি\s*(?:আঁক|বানাও|বানান|চাই)|আঁকো|আঁকে\s?দাও",
    re.IGNORECASE,
)
EDIT_IMG_RE = re.compile(
    r"\b(?:edit|retouch|photoshop)\b|change\s+(?:the\s+)?background|remove\s+(?:the\s+)?background|swap\s+face|এডিট",
    re.IGNORECASE,
)
VIDEO_RE = re.compile(r"\b(?:video|animation|animated\s+clip|motion\s+clip)\b|ভিডিও|অ্যানিমেশন", re.IGNORECASE)
PDF_RE = re.compile(r"\b(?:pdf|cv|resume|document)\b", re.IGNORECASE)
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+")
CODE_RE = re.compile(
    r"```|\b(?:code|function|script|program|debug|traceback|bug|python|javascript|typescript|java|sql|html|css|api|regex|algorithm)\b"
    r"|কোড|প্রোগ্রাম|ফাংশন|এরর",
    re.IGNORECASE,
)
REASON_RE = re.compile(r"\b(?:why|explain|analyze|compare|plan|strategy|prove|derive|reason)\b|কেন|ব্যাখ্যা|বিশ্লেষণ", re.IGNORECASE)


def auto_mode(prompt):
    """Picks the brain mode when the user leaves the sidebar on Auto."""
    text = prompt or ""
    if CODE_RE.search(text):
        return "💻 Code Expert"
    if REASON_RE.search(text) or len(text.split()) >= 45:
        return "🧠 Reasoning"
    return "⚡ Fast"


# --- Basic helpers ---
def secret(name, default=""):
    """Sidebar input wins, then secrets.toml, then env vars."""
    box = st.session_state.get("box_" + name)
    if box and str(box).strip():
        return box
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


def keys_from(value):
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value or "").replace(",", "\n").splitlines() if x.strip()]


def safe_error(exc):
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def extract_urls(text):
    urls = re.findall(r'https?://[^\s<>"\']+', text or "", flags=re.IGNORECASE)
    return [u.rstrip(".,;:!?)\\]}'\"") for u in urls]


def youtube_id(text):
    patterns = (
        r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/(?:shorts|embed|live)/)([A-Za-z0-9_-]{11})",
    )
    for pattern in patterns:
        m = re.search(pattern, text or "")
        if m:
            return m.group(1)
    return None


# --- Web & YouTube tools ---
def unwrap_ddg(href):
    """DuckDuckGo wraps results in //duckduckgo.com/l/?uddg=<encoded>. Unwrap them."""
    if "duckduckgo.com/l/" in (href or ""):
        target = parse_qs(urlparse(href).query).get("uddg", [None])[0]
        if target:
            return unquote(target)
    if href and href.startswith("//"):
        return "https:" + href
    return href or ""


def web_search(query, limit=6):
    try:
        r = HTTP.get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for item in soup.select(".result")[:limit]:
            a = item.select_one(".result__a")
            if not a:
                continue
            snippet_tag = item.select_one(".result__snippet")
            results.append({
                "title": a.get_text(strip=True),
                "url": unwrap_ddg(a.get("href", "")),
                "snippet": snippet_tag.get_text(" ", strip=True) if snippet_tag else "",
            })
        return results
    except Exception:
        return []


def host_is_public(host):
    """Block private/loopback/link-local/reserved targets (SSRF guard)."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except Exception:
        return False
    if not infos:
        return False
    for item in infos:
        try:
            ip = ipaddress.ip_address(item[4][0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def safe_fetch_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS URLs are allowed")
    if not parsed.hostname or not host_is_public(parsed.hostname):
        raise ValueError("Blocked: non-public or unresolvable host")
    # NOTE: requests re-resolves DNS, so a full anti-rebinding fix would also need
    # IP pinning; the byte cap + content-type check + scheme/host guard here stop
    # the practical SSRF file://, localhost, and huge-response vectors.
    with HTTP.get(url, timeout=20, stream=True) as r:
        ctype = (r.headers.get("content-type") or "").lower()
        if not any(t in ctype for t in ("text/", "html", "json", "xml")):
            raise ValueError(f"Unsupported content-type: {ctype or 'unknown'}")
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=65536):
            buf.extend(chunk)
            if len(buf) >= MAX_PAGE_BYTES:
                break
    soup = BeautifulSoup(bytes(buf), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return f"URL: {url}\nContent:\n" + " ".join(soup.get_text(" ", strip=True).split())[:12000]


def get_transcript(video_id):
    # youtube-transcript-api >= 1.x uses the instance API (.fetch); the old static
    # YouTubeTranscriptApi.get_transcript() was removed.
    if YouTubeTranscriptApi:
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en", "bn", "hi"])
            return " ".join(getattr(s, "text", "") or s.get("text", "") for s in fetched)
        except Exception:
            pass
    if UnblockedYT:
        try:
            fetched = UnblockedYT().fetch(video_id)
            return " ".join(getattr(s, "text", "") or s.get("text", "") for s in fetched)
        except Exception:
            pass
    return None


def needs_web(prompt):
    return not youtube_id(prompt) and bool(SEARCH_RE.search(prompt))


# --- File processing ---
def extract_file(upload):
    raw = upload.getvalue()
    name = (upload.name or "").lower()

    if name.endswith(".pdf") and PdfReader:
        try:
            return "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(raw)).pages)[:MAX_CONTEXT_CHARS]
        except Exception as e:
            return f"(PDF read failed: {safe_error(e)})"
    if name.endswith(".docx") and Document:
        try:
            return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)[:MAX_CONTEXT_CHARS]
        except Exception as e:
            return f"(DOCX read failed: {safe_error(e)})"
    if name.endswith((".csv", ".xlsx", ".xls")) and pd:
        try:
            df = pd.read_csv(io.BytesIO(raw)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(raw))
            return (f"TABLE {name}: {df.shape[0]} rows x {df.shape[1]} cols\n"
                    + df.head(40).to_string(index=False))[:MAX_CONTEXT_CHARS]
        except Exception as e:
            return f"(Table read failed: {safe_error(e)})"
    if name.endswith((".txt", ".md", ".py", ".json", ".html", ".xml")):
        return raw.decode("utf-8", errors="replace")[:MAX_CONTEXT_CHARS]
    return ""


# --- AI providers (free-first, automatic rotation) ---
def to_openai_messages(history, image=None):
    messages = [dict(m) for m in history]
    if image:
        data_b64 = base64.b64encode(image[0]).decode()
        uri = f"data:{image[1] or 'image/png'};base64,{data_b64}"
        last = messages[-1]
        messages[-1] = {"role": "user", "content": [
            {"type": "text", "text": last.get("content", "")},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}
    return messages


def _pollinations(messages):
    model = secret("POLLINATIONS_MODEL", DEFAULT_POLLINATIONS_MODEL)
    r = HTTP.post(
        "https://text.pollinations.ai/openai",
        json={"model": model, "messages": messages, "referrer": "jarvis-streamlit"},
        timeout=TIMEOUT,
    )
    if r.status_code == 200:
        try:
            text = r.json()["choices"][0]["message"]["content"]
            if text and text.strip():
                return text
            raise RuntimeError("empty response")
        except (KeyError, IndexError, ValueError):
            return r.text if r.text.strip() else (_ for _ in ()).throw(RuntimeError("empty response"))
    hint = ""
    if r.status_code in (402, 429):
        hint = " — free anonymous quota hit; try again in a minute or add any free key (Groq/OpenRouter/Gemini)"
    raise RuntimeError(f"HTTP {r.status_code}{hint}: {r.text[:150]}")


def _groq_chat(messages, keys):
    model = secret("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    last = None
    for key in keys[:3]:
        try:
            client = Groq(api_key=key)
            comp = client.chat.completions.create(messages=messages, model=model, timeout=TIMEOUT)
            text = comp.choices[0].message.content
            if text and text.strip():
                return text
            last = RuntimeError("empty response")
        except Exception as e:
            last = e
    raise last or RuntimeError("no Groq keys")


def _openrouter_chat(messages, keys):
    models = keys_from(secret("OPENROUTER_MODELS")) or DEFAULT_OPENROUTER_MODELS
    dead_keys, dead_models, last = set(), set(), None
    for key in keys[:3]:
        if key in dead_keys:
            continue
        for model in models:
            if model in dead_models:
                continue
            try:
                r = HTTP.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "messages": messages},
                    timeout=TIMEOUT,
                )
            except Exception as e:
                last = e
                continue
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                if text and text.strip():
                    return text, model
                last = RuntimeError("empty response")
            elif r.status_code in (401, 403):
                dead_keys.add(key)      # dead key → stop trying models with it
                break
            elif r.status_code in (400, 404):
                dead_models.add(model)  # dead model → skip it everywhere
                last = RuntimeError(f"HTTP {r.status_code} for {model}")
            else:
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
    raise last or RuntimeError("no OpenRouter keys")


def _gemini_chat(history, keys, image=None):
    model = secret("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    system_text = "\n".join(m["content"] for m in history if m["role"] == "system")
    contents = []
    for m in history:
        if m["role"] == "system":
            continue
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    if image and contents:
        contents[-1]["parts"].append({
            "inline_data": {"mime_type": image[1] or "image/png",
                            "data": base64.b64encode(image[0]).decode()}
        })
    body = {"contents": contents or [{"role": "user", "parts": [{"text": "Hello"}]}]}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    last = None
    for key in keys[:3]:
        r = HTTP.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": key}, json=body, timeout=TIMEOUT,
        )
        if r.status_code == 200:
            candidates = r.json().get("candidates") or []
            for cand in candidates:
                parts = (cand.get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts)
                if text.strip():
                    return text
            raise RuntimeError("Gemini returned an empty response (possibly safety-blocked)")
        if r.status_code in (400, 404):
            raise RuntimeError(f"Gemini model '{model}' rejected: {r.text[:150]}")
        last = RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
    raise last or RuntimeError("no Gemini keys")


def call_ai(history, image=None, mode="⚡ Fast"):
    """Tries providers in order: free-no-key → Groq → OpenRouter → Gemini."""
    failures = []
    full_history = list(history)
    if full_history and full_history[0].get("role") == "system":
        full_history[0] = dict(full_history[0])
        full_history[0]["content"] += "\n" + MODE_PROMPTS.get(mode, "")
    elif full_history:
        full_history.insert(0, {"role": "system", "content": MODE_PROMPTS.get(mode, "")})

    try:
        answer = _pollinations(to_openai_messages(full_history, image))
        return answer, f"Pollinations ({secret('POLLINATIONS_MODEL', DEFAULT_POLLINATIONS_MODEL)}) — free", failures
    except Exception as e:
        failures.append("Pollinations: " + safe_error(e))

    gq_keys = keys_from(secret("GROQ_KEYS"))
    if gq_keys and Groq:
        msgs = [dict(m) for m in full_history]
        if image:
            msgs.append({"role": "system",
                         "content": "(The user attached an image this turn but this provider is text-only; answer from the text context or say you cannot view it.)"})
        try:
            answer = _groq_chat(msgs, gq_keys)
            return answer, f"Groq ({secret('GROQ_MODEL', DEFAULT_GROQ_MODEL)})", failures
        except Exception as e:
            failures.append("Groq: " + safe_error(e))

    or_keys = keys_from(secret("OPENROUTER_KEYS"))
    if or_keys:
        try:
            answer, model = _openrouter_chat(to_openai_messages(full_history, image), or_keys)
            return answer, f"OpenRouter ({model.split('/')[-1]})", failures
        except Exception as e:
            failures.append("OpenRouter: " + safe_error(e))

    gm_keys = keys_from(secret("GEMINI_KEYS"))
    if gm_keys:
        try:
            answer = _gemini_chat(full_history, gm_keys, image)
            return answer, f"Gemini ({secret('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)})", failures
        except Exception as e:
            failures.append("Gemini: " + safe_error(e))

    offline = ("স্যার, এই মুহূর্তে কোনো এআই প্রোভাইডার সাড়া দিচ্ছে না। "
               "একটি ফ্রি Groq কী (console.groq.com) যোগ করলে উত্তর আরও দ্রুত ও নির্ভরযোগ্য হবে।\n\n"
               "Latest errors:\n" + "\n".join(failures[-3:]))
    return offline, "Offline", failures


# --- Unlimited image generation (free tier needs patient retries) ---
def gen_image(prompt, width=1024, height=1024, seed=0):
    """Returns image bytes, retrying through the free-tier 402/503 window."""
    url = (f"https://image.pollinations.ai/prompt/{quote(prompt)}"
           f"?width={width}&height={height}&nologo=true&model=flux&seed={seed}"
           f"&referrer=jarvis-streamlit")
    last = None
    schedule = (0,) + IMAGE_RETRY_WAITS
    for attempt, wait in enumerate(schedule):
        if wait:
            time.sleep(wait)
        try:
            r = HTTP.get(url, timeout=180)
        except Exception as e:
            last = e
            continue
        if r.status_code == 200 and len(r.content) > 10_000 and "image" in (r.headers.get("content-type") or ""):
            return r.content
        last = RuntimeError(f"HTTP {r.status_code} for image attempt {attempt + 1}")
    raise last or RuntimeError("image generation failed")


# --- AI VIDEO STUDIO engine ---
STORYBOARD_PROMPT = """You are a film director. Break the user's concept into {count} cinematic scenes.
Return ONLY valid JSON, no markdown, exactly this shape:
{{"scenes": [{{"prompt": "<concise ENGLISH image-generation prompt, one scene, visual style: {style}>", "narration": "<one spoken line in the SAME language as the concept>"}}]}}"""


def parse_storyboard(text, count):
    """Extracts the first JSON object; returns a list of scene dicts."""
    if not text:
        return []
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    scenes = data.get("scenes") if isinstance(data, dict) else None
    out = []
    for s in scenes or []:
        if isinstance(s, dict) and str(s.get("prompt", "")).strip():
            out.append({"prompt": str(s["prompt"])[:400], "narration": str(s.get("narration", ""))[:300]})
    return out[:count]


def fallback_storyboard(concept, count, style):
    """Deterministic scenes when the AI storyboard fails — video always completes."""
    angles = ["wide establishing shot", "medium shot with main subject in focus", "dramatic close-up",
              "overhead aerial view", "low-angle hero shot", "golden hour wide shot"]
    return [{"prompt": f"{concept} — {angles[i % len(angles)]}, style: {style}",
             "narration": ""} for i in range(count)]


def scene_filter(zoom_in=True):
    """Ken Burns motion: alternate slow zoom-in / zoom-out per scene."""
    if zoom_in:
        return (f"scale={VIDEO_W}:{VIDEO_H},zoompan=z='min(zoom+0.0020,1.25)'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=120:s={VIDEO_W}x{VIDEO_H}:fps=30,format=yuv420p")
    return (f"scale={VIDEO_W}:{VIDEO_H},zoompan=z='if(eq(on,1),1.25,max(1.0,zoom-0.0020))'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=120:s={VIDEO_W}x{VIDEO_H}:fps=30,format=yuv420p")


def xfade_filter(count, scene_seconds, transition=0.5):
    """Crossfade chain for N pre-rendered clips of equal length."""
    if count < 2:
        return None
    parts, prev = [], "[0:v]"
    for i in range(1, count):
        out = f"[v{i}]" if i < count - 1 else "[v]"
        parts.append(f"{prev}[{i}:v]xfade=transition=fade:duration={transition}:offset={i * (scene_seconds - transition):.2f}{out}")
        prev = out
    return ";".join(parts)


def _run_ffmpeg(args):
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args],
                          capture_output=True, timeout=600)


def render_scene_clip(img_path, out_path, seconds, zoom_in=True):
    args = ["-loop", "1", "-i", img_path,
            "-filter_complex", scene_filter(zoom_in),
            "-t", str(seconds), "-c:v", "libx264", "-preset", "veryfast", out_path]
    return _run_ffmpeg(args).returncode == 0


def title_card_clip(out_path, seconds):
    """Fallback scene (gradient card) so a failed image never blocks the video."""
    args = ["-f", "lavfi", "-i", f"gradients=s={VIDEO_W}x{VIDEO_H}:c0=0x101a2c:c1=0x1a2d4a:speed=0.02",
            "-t", str(seconds), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out_path]
    return _run_ffmpeg(args).returncode == 0


def concat_clips(clip_paths, out_path, scene_seconds):
    if not clip_paths:
        return False
    if len(clip_paths) == 1:
        return _run_ffmpeg(["-i", clip_paths[0], "-c:v", "libx264", "-preset", "veryfast",
                            "-movflags", "+faststart", out_path]).returncode == 0
    filt = xfade_filter(len(clip_paths), scene_seconds)
    args = []
    for p in clip_paths:
        args += ["-i", p]
    args += ["-filter_complex", filt, "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast",
             "-movflags", "+faststart", out_path]
    return _run_ffmpeg(args).returncode == 0


def mux_narration(video_path, audio_bytes, out_path):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fh:
        fh.write(audio_bytes)
        audio_path = fh.name
    try:
        ok = _run_ffmpeg(["-i", video_path, "-i", audio_path, "-map", "0:v", "-map", "1:a",
                          "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
                          out_path]).returncode == 0
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
    return ok


def make_narration(text, voice_name):
    audio = bytearray()

    async def _tts():
        async for chunk in edge_tts.Communicate(text, voice_name).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])

    asyncio.run(_tts())
    return bytes(audio)


# --- PDF export (Bengali-capable, shaped, emoji-safe) ---
def strip_for_pdf(text):
    text = EMOJI_RE.sub("", text or "")
    text = re.sub(r"[*_`#>]+", "", text)
    return text.strip()


def ensure_bengali_font():
    path = os.path.join(tempfile.gettempdir(), "NotoSansBengali-Regular.ttf")
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    for url in BENGALI_FONT_URLS:
        try:
            r = HTTP.get(url, timeout=90)
            if r.status_code == 200 and len(r.content) > 100_000:
                if url.endswith(".zip"):
                    import zipfile
                    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                        for member in zf.namelist():
                            if member.endswith("NotoSansBengali-Regular.ttf"):
                                with open(path, "wb") as fh:
                                    fh.write(zf.read(member))
                                return path
                    continue
                with open(path, "wb") as fh:
                    fh.write(r.content)
                return path
        except Exception:
            continue
    return None


def build_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    clean = strip_for_pdf(text)
    font_path = ensure_bengali_font()
    if font_path:
        try:
            pdf.add_font("NotoBengali", "", font_path)
            pdf.set_font("NotoBengali", size=11)
            try:
                pdf.set_text_shaping(True)  # correct Bengali conjuncts (needs uharfbuzz)
            except Exception:
                pass
        except Exception:
            pdf.set_font("Helvetica", size=11)
            clean = clean.encode("latin-1", "replace").decode("latin-1")
    else:
        pdf.set_font("Helvetica", size=11)
        clean = clean.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 7, clean)
    return bytes(pdf.output())


# --- Deep Research pipeline ---
def deep_research(prompt, status, failures):
    """Multi-query search + page reading; returns (context, sources)."""
    context, sources, seen = [], [], set()
    queries = [prompt, f"{prompt} explained", f"{prompt} latest 2026"]
    status.update(label="🌐 Deep research: running 3 searches…")
    results = []
    for q in queries:
        for r in web_search(q, limit=5):
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    results = results[:6]
    if not results:
        failures.append("deep research: no search results")
        return context, sources
    sources = [(r["title"], r["url"]) for r in results]
    context.append("WEB RESULTS:\n" + "\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}" for i, r in enumerate(results, 1)))
    for i, result in enumerate(results[:3], 1):
        status.update(label=f"📖 Reading source [{i}]…")
        try:
            page = safe_fetch_url(result["url"])
            context.append(f"SOURCE [{i}] {result['title']} — {result['url']}\n{page[:6000]}")
        except Exception as e:
            failures.append(f"page reader [{i}]: " + safe_error(e))
    return context, sources


# --- Session state & free local persistence ---
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chat_history.json")


def persist_messages():
    """Keeps chat history across page refreshes — no database needed."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(st.session_state.messages[-50:], fh, ensure_ascii=False)
    except Exception:
        pass


def init_session_state():
    st.session_state.setdefault("last_audio_id", None)
    st.session_state.setdefault("pending_prompt", None)
    if "messages" not in st.session_state:
        loaded = []
        try:
            with open(HISTORY_FILE, encoding="utf-8") as fh:
                loaded = json.load(fh)
        except Exception:
            loaded = []
        st.session_state.messages = loaded if isinstance(loaded, list) else []


# --- Main app ---
def main():
    st.set_page_config(page_title="JARVIS Ultimate", page_icon="🤖", layout="wide")
    init_session_state()

    st.markdown("""
<style>
.stApp { background: #0b1020; color: #eef4ff; }
[data-testid="stSidebar"] { background: #101a2c; }
[data-testid="stChatMessage"] { border: 1px solid #263750; border-radius: 15px; padding: 12px; background: #151f33; }
.jv-pill { color: #5ee7ff; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.1em; background: #1a2d4a; padding: 5px 10px; border-radius: 20px; display: inline-block;}
</style>
""", unsafe_allow_html=True)

    with st.sidebar:
        st.title("🤖 JARVIS")
        st.markdown(f'<div class="jv-pill">VERSION {APP_VERSION}</div>', unsafe_allow_html=True)
        st.caption("ChatGPT/Gemini/DeepSeek-style abilities — free: multi-mode brain, deep research, "
                   "vision, files, unlimited image gen, AI video studio, PDF, voice.")
        st.divider()

        st.subheader("🧠 Brain mode")
        mode_choice = st.selectbox("Mode", ["🤖 Auto (smart routing)"] + list(MODE_PROMPTS.keys()), index=0)
        deep_research_on = st.toggle("🌐 Deep Research (multi-source, cited answers)", value=False)

        st.subheader("⚡ Quick actions")
        for label, ready_prompt in [
            ("📰 Today's top tech news (cited)", "Search today's top technology news and summarize the top 5 items with sources."),
            ("🎨 Generate: futuristic Dhaka skyline", "Generate an image of a futuristic Dhaka skyline at sunset, ultra detailed"),
            ("🎥 Video: tigers of the Sundarbans", "Make a short video about the tigers of the Sundarbans"),
            ("📄 Make my CV (PDF)", "Create a professional ATS-friendly CV for a fresh graduate in computer science with strong Python and Streamlit skills, and give me the PDF."),
        ]:
            if st.button(label, use_container_width=True):
                st.session_state.pending_prompt = ready_prompt
                st.rerun()

        if st.session_state.messages:
            md = "\n\n".join(
                (f"**🧑 {m['content']}**" if m["role"] == "user"
                 else f"**🤖 JARVIS** ({m.get('provider', '')})\n\n{m['content']}")
                for m in st.session_state.messages
            )
            st.download_button("💾 Download chat (.md)", data=md, file_name="JARVIS_chat.md", mime="text/markdown")

        st.subheader("🎬 Video Studio")
        with st.expander("Video settings"):
            vid_scenes = st.slider("Scenes", 2, 6, 3)
            vid_seconds = st.slider("Seconds per scene", 2, 6, 4)
            vid_style = st.selectbox("Style", ["cinematic film still", "anime art", "3D render",
                                               "documentary photo", "watercolor painting"])
            vid_narration = st.toggle("Narrate with AI voice", value=False)

        st.divider()
        st.subheader("🔑 API keys (optional)")
        st.caption("All optional — Pollinations works with no key. Multiple keys (comma/newline) rotate automatically.")
        st.text_area("OpenRouter keys", key="box_OPENROUTER_KEYS", type="password", height=68)
        st.text_area("Groq keys", key="box_GROQ_KEYS", type="password", height=68)
        st.text_area("Gemini keys", key="box_GEMINI_KEYS", type="password", height=68)
        with st.expander("🔌 Providers & honest limits"):
            st.markdown("🟢 **Pollinations** — free, NO key (default brain + images + video scenes)")
            st.markdown(f"{'🟢' if keys_from(secret('GROQ_KEYS')) else '⚪'} **Groq** — free key: console.groq.com")
            st.markdown(f"{'🟢' if keys_from(secret('OPENROUTER_KEYS')) else '⚪'} **OpenRouter** — free key: openrouter.ai")
            st.markdown(f"{'🟢' if keys_from(secret('GEMINI_KEYS')) else '⚪'} **Gemini** — free key: aistudio.google.com")
            st.info("**Honest limits:** free hosted LLMs are rate-limited (the app auto-retries & falls back). "
                    "Image/video generation is unlimited but needs patience on the free tier (built-in retries). "
                    "Veo/Flow-grade generative video is a paid product — JARVIS renders real MP4s from generated "
                    "scenes instead, which *is* unlimited and free.")

        st.divider()
        st.subheader("🔊 Voice (TTS)")
        voice_out = st.toggle("Enable spoken replies", value=False)
        voice_name = st.selectbox("Voice", ["bn-BD-PradeepNeural", "bn-BD-NabanitaNeural",
                                            "en-US-GuyNeural", "en-US-AriaNeural"])

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("JARVIS Ultimate")
    st.caption("Research • code • vision • files • images • videos • PDFs • voice — grounded in live web sources.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("provider"):
                st.caption(f"⚡ {msg['provider']}")
            for i, (title, url) in enumerate(msg.get("sources") or [], 1):
                st.markdown(f"{i}. [{title}]({url})")

    upload = st.file_uploader(
        "Attach a file or image (optional)",
        type=["pdf", "docx", "txt", "md", "py", "json", "csv", "xlsx", "png", "jpg", "jpeg", "webp"],
    )
    is_image_upload = bool(upload) and bool(upload.type or "").startswith("image/")

    if mic_recorder and st.toggle("🎙️ Voice input", value=False):
        recorded = mic_recorder(start_prompt="Record", stop_prompt="Stop", just_once=True)
        if recorded and recorded.get("bytes"):
            rid = hashlib.sha1(recorded["bytes"]).hexdigest()
            if rid != st.session_state.last_audio_id:
                st.session_state.last_audio_id = rid
                gq_keys = keys_from(secret("GROQ_KEYS"))
                if Groq and gq_keys:
                    try:
                        client = Groq(api_key=gq_keys[0])
                        res = client.audio.transcriptions.create(
                            file=("voice.wav", recorded["bytes"]), model="whisper-large-v3-turbo")
                        st.session_state.pending_prompt = res.text
                        st.rerun()
                    except Exception as e:
                        st.error(safe_error(e))

    prompt = st.session_state.pending_prompt or st.chat_input("Ask, research, upload, or say “make a video about…”…")
    if st.session_state.pending_prompt:
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        persist_messages()
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("JARVIS is working…", expanded=False)
            context = []
            sources = []
            tools_used = []
            failures = []
            image = (upload.getvalue(), upload.type or "image/png") if is_image_upload else None

            try:
                # 1) YouTube transcript
                vid = youtube_id(prompt)
                if vid:
                    status.update(label="🎬 Fetching YouTube transcript…")
                    transcript = get_transcript(vid)
                    if transcript:
                        context.append(f"YOUTUBE TRANSCRIPT:\n{transcript[:MAX_CONTEXT_CHARS]}")
                        tools_used.append("YouTube transcript")
                    else:
                        context.append(f"YOUTUBE: transcript unavailable for video {vid}.")
                        failures.append("YouTube transcript fetch failed")

                # 2) Read a link the user pasted
                elif "http" in prompt:
                    status.update(label="📖 Reading the pasted link…")
                    urls = extract_urls(prompt)
                    if urls:
                        try:
                            context.append(safe_fetch_url(urls[0]))
                            tools_used.append("page reader")
                        except Exception as e:
                            context.append("Website fetch failed.")
                            failures.append("page reader: " + safe_error(e))

                # 3) Deep research or normal search
                if not vid and "http" not in prompt and (deep_research_on or needs_web(prompt) or MEDIA_RE.search(prompt)):
                    if deep_research_on:
                        tools_used.append("deep research (3 queries + page reading)")
                        rc, rs = deep_research(prompt, status, failures)
                        context.extend(rc)
                        sources.extend(rs)
                    else:
                        media = bool(MEDIA_RE.search(prompt))
                        status.update(label="🎬 Finding where to watch…" if media else "🔍 Searching the web…")
                        query = f"{prompt} official watch online" if media else prompt
                        results = web_search(query)
                        if results:
                            sources = [(r["title"], r["url"]) for r in results]
                            context.append("WEB RESULTS:\n" + "\n".join(
                                f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}"
                                for i, r in enumerate(results, 1)))
                            tools_used.append("web search")
                            if not media:
                                status.update(label="📖 Reading the top sources…")
                                for i, result in enumerate(results[:2], 1):
                                    try:
                                        page = safe_fetch_url(result["url"])
                                        context.append(f"SOURCE [{i}] {result['title']} — {result['url']}\n{page[:6000]}")
                                    except Exception as e:
                                        failures.append(f"page reader [{i}]: " + safe_error(e))
                                tools_used.append("page reader")
                        else:
                            failures.append("web search returned no results")

                # 4) AI VIDEO STUDIO — chat-triggered
                video_done = False
                if VIDEO_RE.search(prompt) and not image:
                    status.update(label="🎥 Video Studio: writing the storyboard…")
                    tools_used.append("AI video studio")
                    count = max(2, min(vid_scenes, 6))
                    sb_raw, sb_prov, sb_fail = call_ai(
                        [{"role": "system", "content": "You are a film director."},
                         {"role": "user", "content": STORYBOARD_PROMPT.format(count=count, style=vid_style) + "\n\nConcept: " + prompt}],
                        mode="⚡ Fast")
                    scenes = parse_storyboard(sb_raw, count) or fallback_storyboard(prompt, count, vid_style)
                    context.append("VIDEO STORYBOARD:\n" + "\n".join(f"{i+1}. {s['prompt']}" for i, s in enumerate(scenes)))
                    if sb_fail:
                        failures.extend(sb_fail)

                    workdir = tempfile.mkdtemp(prefix="jarvis_video_")
                    clips = []
                    status.update(label=f"🎨 Video Studio: generating {len(scenes)} scenes in parallel… (free-tier retries are normal)")

                    def _scene_image(i, scene):
                        try:
                            return gen_image(scene["prompt"], VIDEO_W, VIDEO_H, seed=100 + i)
                        except Exception as e:
                            failures.append(f"video scene {i + 1}: " + safe_error(e) + " → using gradient card")
                            return None

                    with ThreadPoolExecutor(max_workers=min(4, len(scenes))) as pool:
                        images = list(pool.map(_scene_image, range(len(scenes)), scenes))

                    for i, img in enumerate(images):
                        img_path = os.path.join(workdir, f"scene_{i}.jpg")
                        if img:
                            with open(img_path, "wb") as fh:
                                fh.write(img)
                        if not os.path.exists(img_path):
                            grad_clip = os.path.join(workdir, f"clip_{i}.mp4")
                            if title_card_clip(grad_clip, vid_seconds):
                                clips.append(grad_clip)
                        else:
                            clip_path = os.path.join(workdir, f"clip_{i}.mp4")
                            if render_scene_clip(img_path, clip_path, vid_seconds, zoom_in=(i % 2 == 0)):
                                clips.append(clip_path)

                    if clips:
                        status.update(label="🎞️ Video Studio: stitching scenes…")
                        silent_path = os.path.join(workdir, "silent.mp4")
                        if concat_clips(clips, silent_path, vid_seconds):
                            final_path = silent_path
                            if vid_narration and edge_tts:
                                try:
                                    narration = " ".join(s.get("narration", "") for s in scenes if s.get("narration")).strip()
                                    if narration:
                                        audio = make_narration(narration[:3000], voice_name)
                                        voiced_path = os.path.join(workdir, "final.mp4")
                                        if mux_narration(silent_path, audio, voiced_path):
                                            final_path = voiced_path
                                            tools_used.append("AI narration")
                                except Exception as e:
                                    failures.append("narration: " + safe_error(e))
                            with open(final_path, "rb") as fh:
                                video_bytes = fh.read()
                            st.video(video_bytes)
                            st.download_button("📥 Download video (MP4)", data=video_bytes,
                                               file_name="JARVIS_video.mp4", mime="video/mp4")
                            video_done = True
                        else:
                            failures.append("video stitching failed")
                    else:
                        failures.append("video: no scenes could be rendered")

                # 5) Image generation / honest edit warning
                if GEN_IMG_RE.search(prompt) and not image:
                    status.update(label="🎨 Generating image… (free tier retries automatically)")
                    tools_used.append("image generation")
                    try:
                        img_bytes = gen_image(prompt, 1024, 1024, seed=abs(hash(prompt)) % 100000)
                        st.image(img_bytes, caption="Generated by JARVIS (Pollinations, free)")
                        st.download_button("📥 Download image", data=img_bytes,
                                           file_name="JARVIS_image.jpg", mime="image/jpeg")
                        context.append("An image was generated for the user via Pollinations.")
                    except Exception as e:
                        failures.append("image gen: " + safe_error(e))
                        st.warning("⚠️ ছবি জেনারেশন এখন ফ্রি কোটায় ব্যস্ত — ১ মিনিট পর আবার চেষ্টা করুন, বা ফ্রি কী যোগ করুন।")
                if image and EDIT_IMG_RE.search(prompt):
                    st.warning("⚠️ সতর্কতা: ফ্রি এআই জেনারেটর দিয়ে হুবহু চেহারা ঠিক রেখে ছবি এডিট করা সম্ভব নয় — এটি নতুন একটি ছবি তৈরি করবে। হুবহু এডিটের জন্য প্রফেশনাল টুল (যেমন Remaker AI, Pincel) লাগবে।")

                # 6) Attached files
                if upload and not is_image_upload:
                    extracted = extract_file(upload)
                    if extracted:
                        context.append(f"FILE ({upload.name}):\n{extracted}")
                        tools_used.append(f"file reader ({upload.name})")

                # 7) Build the enriched prompt + recent history
                enriched = prompt
                if context:
                    enriched += "\n\n--- CONTEXT (real tool results) ---\n" + "\n\n".join(context)
                tail = st.session_state.messages[-8:]
                history = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                    {"role": m["role"], "content": m["content"]} for m in tail
                ]
                history[-1]["content"] = enriched

                # 8) Get the answer (free provider chain)
                effective_mode = auto_mode(prompt) if mode_choice.startswith("🤖") else mode_choice
                status.update(label=f"🧠 Thinking… ({effective_mode.strip('⚡🧠💻 ')})")
                answer, provider, call_failures = call_ai(history, image, effective_mode)
                failures.extend(call_failures)
                status.update(label="Done", state="complete")

                st.markdown(answer)
                st.caption(f"⚡ {provider} · {effective_mode}" + (f" · 🧰 {', '.join(tools_used)}" if tools_used else ""))
                for i, (title, url) in enumerate(sources, 1):
                    st.markdown(f"{i}. [{title}]({url})")
                if video_done:
                    context_note = "The video was rendered above and can be downloaded."
                    answer += f"\n\n🎥 {context_note}"

                # Persist the reply BEFORE optional extras, so a crash can never eat it
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "provider": provider, "sources": sources,
                })
                st.session_state.messages = st.session_state.messages[-50:]
                persist_messages()

                # Auto PDF (Bengali-safe)
                if PDF_RE.search(prompt) and FPDF:
                    try:
                        data = build_pdf(answer)
                        st.download_button("📥 Download PDF", data=data, file_name="JARVIS.pdf", mime="application/pdf")
                    except Exception as e:
                        failures.append("pdf: " + safe_error(e))
                        st.error("PDF export failed: " + safe_error(e))

                # Voice output
                if voice_out and provider != "Offline" and edge_tts:
                    try:
                        audio = make_narration(answer[:3000], voice_name)
                        st.audio(audio, format="audio/mp3")
                    except Exception as e:
                        failures.append("tts: " + safe_error(e))

                if failures:
                    with st.expander("🛠 Tool & provider details (debug)"):
                        st.text("\n".join(failures))

            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"JARVIS encountered an error: {safe_error(e)}")


if __name__ == "__main__":
    main()
