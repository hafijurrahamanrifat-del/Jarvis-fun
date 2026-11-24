# JARVIS ULTIMATE FINAL — All-in-One AI Agent for Streamlit
# Features: Multi-AI Router, Media Search, Image Gen, Auto PDF, YouTube, Web Scraping, Voice.

from __future__ import annotations
import os, re, io, json, time, hashlib, ipaddress, socket, asyncio
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin, quote

import requests
import streamlit as st
from bs4 import BeautifulSoup

# --- Safe Optional Imports ---
try: from groq import Groq
except Exception: Groq = None
try: from youtube_transcript_api import YouTubeTranscriptApi
except Exception: YouTubeTranscriptApi = None
try: from youtube_transcript_unblocked import YouTubeTranscriptApi as UnblockedYT
except Exception: UnblockedYT = None
try: from streamlit_mic_recorder import mic_recorder
except Exception: mic_recorder = None
try: import edge_tts
except Exception: edge_tts = None
try: from pypdf import PdfReader
except Exception: PdfReader = None
try: from docx import Document
except Exception: Document = None
try: import pandas as pd
except Exception: pd = None
try: from fpdf import FPDF
except Exception: FPDF = None
try: import google.generativeai as genai
except Exception: genai = None

APP_VERSION = "3.0.0 Final"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "JARVIS-Ultimate/3.0"})
TIMEOUT = 50
MAX_CONTEXT_CHARS = 24000

# --- The Ultimate System Prompt ---
SYSTEM_PROMPT = """You are JARVIS, an elite, highly capable personal AI assistant and mentor. 
Your capabilities: Expert Coding, ATS CV creation, Deep Research, and Media Search Agent.

CRITICAL RULES:
1. MEDIA SEARCH: If the user wants to watch a movie/anime (especially Hindi Dub), search your knowledge and provide 2-3 direct links to streaming sites (like AnimeHeaven, MovieKelce). Warn them to use an AdBlocker.
2. IMAGE EDITING WARNING: If the user uploads an image and asks to "edit" it while keeping the face EXACTLY the same (e.g., changing background, clothes), you MUST warn them: "স্যার, আমাদের ফ্রি জেনারেটর দিয়ে হুবহু চেহারা ঠিক রেখে এডিট করা সম্ভব নয়। তবে আমি আপনাকে কিছু ফ্রি সাইট দিচ্ছি যেখানে এটি করা যায়।" Then, recommend tools like [Remaker AI](https://remaker.ai/), [Pincel](https://pincel.app/), or [Leonardo AI](https://leonardo.ai/).
3. TOOL RECOMMENDATIONS: When providing solutions needing external tools, provide links in this priority: 
   - 🟢 100% Free & Unlimited (e.g., CapCut, Pollinations AI)
   - 🟡 Free Credits/Trial Sites
   - 🟣 New & Trending Tools
4. SAFETY: Warn the user that auto-logging into external sites via bots violates TOS and can get their Gmail banned. Provide links and prompts instead.
5. Always answer in the user's language (Bengali or English). Format beautifully with emojis.
6. If the user asks for a PDF or CV, format your response professionally. A download button will be generated automatically."""

st.set_page_config(page_title="JARVIS Ultimate", page_icon="🤖", layout="wide")

# --- Premium CSS ---
st.markdown("""
<style>
.stApp { background: #0b1020; color: #eef4ff; }
[data-testid="stSidebar"] { background: #101a2c; }
[data-testid="stChatMessage"] { border: 1px solid #263750; border-radius: 15px; padding: 12px; background: #151f33; }
.jv-pill { color: #5ee7ff; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.1em; background: #1a2d4a; padding: 5px 10px; border-radius: 20px; display: inline-block;}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def secret(name, default=""):
    try: return st.secrets.get(name, default)
    except: return os.getenv(name, default)

def keys_from(value):
    if isinstance(value, (list, tuple)): return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value or "").replace(",", "\n").splitlines() if x.strip()]

def safe_error(exc): return f"{type(exc).__name__}: {str(exc)[:200]}"

if "messages" not in st.session_state: st.session_state.messages = []
if "last_audio_id" not in st.session_state: st.session_state.last_audio_id = None
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None

# --- Web & YouTube Tools ---
def extract_urls(text): return re.findall(r'https?://[^\s<>"\']+', text or "", flags=re.I)

def youtube_id(text):
    for p in [r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})", r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"]:
        m = re.search(p, text)
        if m: return m.group(1)
    return None

def get_transcript(video_id):
    if YouTubeTranscriptApi:
        try:
            rows = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "bn", "hi"])
            return " ".join([t['text'] for t in rows])
        except: pass
    if UnblockedYT:
        try:
            api = UnblockedYT()
            fetched = api.fetch(video_id)
            return " ".join([str(x.get("text", "")) for x in fetched])
        except: pass
    return None

def host_is_public(host):
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return bool(infos) and all((lambda ip: not (ip.is_private or ip.is_loopback))(ipaddress.ip_address(item[4][0])) for item in infos)
    except: return False

def safe_fetch_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not host_is_public(parsed.hostname): raise ValueError("Invalid URL")
    r = HTTP.get(url, timeout=20, stream=True)
    soup = BeautifulSoup(r.raw.read(2_000_000), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]): tag.decompose()
    return f"URL: {url}\nContent:\n" + " ".join(soup.get_text(" ", strip=True).split())[:14000]

def web_search(query, limit=5):
    try:
        r = HTTP.get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        return [{"title": a.get_text(strip=True), "url": a["href"]} for item in soup.select(".result")[:limit] if (a := item.select_one(".result__a"))]
    except: return []

# --- File Processing ---
def extract_file(upload):
    raw = upload.getvalue()
    if upload.name.endswith(".pdf") and PdfReader:
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)[:MAX_CONTEXT_CHARS]
    if upload.name.endswith(".docx") and Document:
        return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)[:MAX_CONTEXT_CHARS]
    if upload.name.endswith((".txt", ".md", ".py", ".json")):
        return raw.decode("utf-8", errors="replace")[:MAX_CONTEXT_CHARS]
    return ""

# --- The AI Router ---
def call_ai(messages, image_parts=None):
    failures = []
    
    # 1. OpenRouter (DeepSeek, Gemini, Llama)
    or_keys = keys_from(secret("OPENROUTER_KEYS", ""))
    if or_keys:
        models = ["deepseek/deepseek-r1:free", "google/gemini-2.5-pro-exp:free", "meta-llama/llama-3.3-70b-instruct:free"]
        for key in or_keys:
            for model in models:
                try:
                    res = HTTP.post("https://openrouter.ai/api/v1/chat/completions", 
                        headers={"Authorization": f"Bearer {key}", "HTTP-Referer": "https://streamlit.app"},
                        json={"model": model, "messages": messages}, timeout=TIMEOUT)
                    if res.status_code == 200:
                        return res.json()['choices'][0]['message']['content'], f"OpenRouter ({model.split('/')[-1]})"
                except Exception as e: failures.append(safe_error(e))
                
    # 2. Groq
    gq_keys = keys_from(secret("GROQ_KEYS", ""))
    if gq_keys and Groq:
        for key in gq_keys:
            try:
                client = Groq(api_key=key)
                comp = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
                return comp.choices[0].message.content, "Groq (Llama 3.3)"
            except Exception as e: failures.append(safe_error(e))
            
    # 3. Gemini Direct
    gm_keys = keys_from(secret("GEMINI_KEYS", ""))
    if gm_keys and genai:
        for key in gm_keys:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                prompt_text = "\n".join([m['content'] for m in messages if m['role'] != 'system'])
                response = model.generate_content(prompt_text)
                return response.text, "Gemini 2.0"
            except Exception as e: failures.append(safe_error(e))
            
    return f"স্যার, সব এআই সার্ভার ব্যস্ত। Error: {failures[-1] if failures else 'Keys missing'}", "Offline"

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 JARVIS")
    st.markdown(f'<div class="jv-pill">VERSION {APP_VERSION}</div>', unsafe_allow_html=True)
    st.divider()
    
    st.subheader("🔑 API Credentials")
    st.text_area("OpenRouter Keys", value=str(secret("OPENROUTER_KEYS", "")), height=68)
    st.text_area("Groq Keys", value=str(secret("GROQ_KEYS", "")), height=68)
    st.text_area("Gemini Keys", value=str(secret("GEMINI_KEYS", "")), height=68)
    
    st.divider()
    st.subheader("🔊 Voice (TTS)")
    voice_out = st.toggle("Enable Spoken Replies", value=False)
    voice_name = st.selectbox("Voice", ["bn-BD-PradeepNeural", "en-US-GuyNeural", "bn-BD-NabanitaNeural"])
    
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main UI ---
st.title("JARVIS Ultimate")
st.caption("Your elite AI mentor, coder, researcher, and media agent.")

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "provider" in msg: st.caption(f"⚡ {msg['provider']}")

# Upload & Voice
upload = st.file_uploader("Attach a file or image (optional)", type=["pdf", "docx", "txt", "png", "jpg"])
if mic_recorder and st.toggle("🎙️ Voice Input", value=False):
    recorded = mic_recorder(start_prompt="Record", stop_prompt="Stop", just_once=True)
    if recorded and recorded.get("bytes"):
        rid = hashlib.sha1(recorded["bytes"]).hexdigest()
        if rid != st.session_state.last_audio_id:
            st.session_state.last_audio_id = rid
            if Groq and keys_from(secret("GROQ_KEYS", "")):
                try:
                    client = Groq(api_key=keys_from(secret("GROQ_KEYS", ""))[0])
                    res = client.audio.transcriptions.create(file=("voice.wav", recorded["bytes"]), model="whisper-large-v3-turbo")
                    st.session_state.pending_prompt = res.text
                    st.rerun()
                except Exception as e: st.error(safe_error(e))

# Chat Input
prompt = st.session_state.pending_prompt or st.chat_input("Message JARVIS...")
if st.session_state.pending_prompt: st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        status = st.status("JARVIS is working...", expanded=False)
        context = []
        
        try:
            # 1. Image Gen & Edit Warning
            image_keywords = ["image", "draw", "picture", "ছবি", "আঁক", "cartoon", "edit"]
            if any(word in prompt.lower() for word in image_keywords) or (upload and upload.type.startswith("image/")):
                if upload and upload.type.startswith("image/"):
                    st.warning("⚠️ সতর্কতা: ফ্রি এআই জেনারেটর দিয়ে আপনার হুবহু চেহারা ঠিক রেখে ছবি এডিট করা সম্ভব নয় (এটি একটি নতুন ছবি তৈরি করবে)।")
                    
                status.update(label="Generating Image...")
                img_url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width=1024&height=1024&nologo=true"
                st.image(img_url, caption="Generated by JARVIS")
                context.append("Image generated successfully.")

            # 2. YouTube Link
            vid = youtube_id(prompt)
            if vid:
                status.update(label="Fetching YouTube...")
                transcript = get_transcript(vid)
                if transcript: context.append(f"YOUTUBE TRANSCRIPT:\n{transcript[:MAX_CONTEXT_CHARS]}")
                else: context.append("YouTube fetch failed.")
            
            # 3. Web Search (If prompt asks for movies/sites or has URL)
            elif "http" in prompt and not vid:
                status.update(label="Reading Website...")
                try: context.append(safe_fetch_url(extract_urls(prompt)[0].rstrip(".,);]")))
                except: context.append("Failed to read website.")
            
            # 4. Media Search or General Search
            elif any(w in prompt.lower() for w in ["movie", "anime", "series", "hindi", "সিনেমা", "অ্যানিমে", "link"]):
                status.update(label="Searching the web for media...")
                results = web_search(prompt)
                if results:
                    context.append("WEB RESULTS:\n" + "\n".join([f"- {r['title']}\n{r['url']}" for r in results]))

            # 5. File Upload
            if upload and not upload.type.startswith("image/"):
                extracted = extract_file(upload)
                context.append(f"FILE ({upload.name}):\n{extracted}")

            # Build Final Prompt
            enriched = prompt
            if context: enriched += "\n\n--- CONTEXT ---\n" + "\n\n".join(context)
            
            history = [{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-5:]]
            history[-1]["content"] = enriched
            
            status.update(label="Consulting AI Providers...")
            answer, provider = call_ai(history)
            status.update(label="Done!", state="complete")
            
            st.markdown(answer)
            st.caption(f"⚡ {provider}")
            
            # Auto PDF Generator
            if any(w in prompt.lower() for w in ["pdf", "cv", "document"]):
                if FPDF:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=11)
                    pdf.multi_cell(0, 7, answer.encode('latin-1', 'replace').decode('latin-1'))
                    st.download_button("📥 Download PDF", pdf.output(dest='S').encode('latin-1'), file_name="JARVIS.pdf", mime="application/pdf")

            # Voice Output
            if voice_out and provider != "Offline" and edge_tts:
                audio = bytearray()
                async def _tts(): 
                    async for chunk in edge_tts.Communicate(answer[:1000], voice_name).stream():
                        if chunk["type"] == "audio": audio.extend(chunk["data"])
                asyncio.run(_tts())
                st.audio(audio, format="audio/mp3")

            st.session_state.messages.append({"role": "assistant", "content": answer, "provider": provider})
            st.session_state.messages = st.session_state.messages[-50:]
            
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"JARVIS encountered an error: {safe_error(e)}")
