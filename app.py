import asyncio
from datetime import datetime
import io
import os
import re
import sqlite3
from bs4 import BeautifulSoup
from ddgs import DDGS
import edge_tts
from fpdf import FPDF
from google import genai
from groq import Groq
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
import qrcode
import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_unblocked import YouTubeTranscriptApi

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="JARVIS Ultimate", page_icon="🤖", layout="wide")

# --- CUSTOM CSS THEME ---
st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; color: #00d2ff; }
    h1, h2, h3 { color: #00d2ff !important; }
    .stButton>button { background-color: #00d2ff; color: black; border-radius: 5px; font-weight: bold; }
    .stTextArea textarea { background-color: #1e1e1e; color: white; }
    .stTextInput input { background-color: #1e1e1e; color: white; }
</style>
""",
    unsafe_allow_html=True,
)

# Docker volume থাকলে /app/data/jarvis.db ব্যবহার করবে, নাহলে লোকাল jarvis.db
DB_DIR = "/app/data" if os.path.isdir("/app/data") else "."
DB_PATH = os.path.join(DB_DIR, "jarvis.db")


# --- DATABASE SETUP & HELPERS ---
@st.cache_resource
def init_db():
  conn = sqlite3.connect(DB_PATH, check_same_thread=False)
  try:
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            provider TEXT,
            created_at TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS idea_vault (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea TEXT NOT NULL,
            created_at TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS finance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            date TEXT NOT NULL
        )"""
    )
    conn.commit()
  finally:
    conn.close()
  return True


init_db()


def get_conn():
  return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_chat_history(limit: int = 40) -> list[dict]:
  conn = get_conn()
  try:
    rows = conn.execute(
        "SELECT role, content, provider FROM chat_history ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    rows.reverse()
    return [
        {"role": r[0], "content": r[1], "provider": r[2] or ""} for r in rows
    ]
  finally:
    conn.close()


def save_chat_message(role: str, content: str, provider: str = ""):
  conn = get_conn()
  try:
    conn.execute(
        "INSERT INTO chat_history (role, content, provider, created_at) VALUES (?, ?, ?, ?)",
        (role, content, provider, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
  finally:
    conn.close()


def clear_chat_history():
  conn = get_conn()
  try:
    conn.execute("DELETE FROM chat_history")
    conn.commit()
  finally:
    conn.close()


# --- CACHED LIVE FEEDS (FREE) ---
@st.cache_data(ttl=600)
def fetch_usd_bdt() -> float:
  res = requests.get(
      "https://api.exchangerate-api.com/v4/latest/USD", timeout=10
  )
  res.raise_for_status()
  return float(res.json()["rates"]["BDT"])


@st.cache_data(ttl=600)
def fetch_crypto_prices() -> dict:
  url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
  res = requests.get(url, timeout=10)
  res.raise_for_status()
  return res.json()


@st.cache_data(ttl=600)
def fetch_tech_news(max_results: int = 4):
  return DDGS().news(keywords="technology AI", max_results=max_results)


# --- FREE TEXT-TO-SPEECH (EDGE-TTS) ---
async def _generate_tts_async(text: str, voice: str) -> bytes:
  clean_text = re.sub(r"[*#`>_~\[\]()]", " ", text)
  clean_text = re.sub(r"\s+", " ", clean_text).strip()[:1200]
  communicate = edge_tts.Communicate(clean_text, voice)
  audio_chunks = bytearray()
  async for chunk in communicate.stream():
    if chunk["type"] == "audio":
      audio_chunks.extend(chunk["data"])
  return bytes(audio_chunks)


def synthesize_voice(
    text: str, voice: str = "en-GB-RyanNeural"
) -> bytes | None:
  try:
    return asyncio.run(_generate_tts_async(text, voice))
  except Exception:
    return None


# --- FREE YOUTUBE & WEB SCRAPER ---
def extract_youtube_id(url: str) -> str | None:
  patterns = [
      r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
      r"youtu\.be\/([0-9A-Za-z_-]{11})",
  ]
  for pat in patterns:
    match = re.search(pat, url.strip())
    if match:
      return match.group(1)
  return None


def get_youtube_transcript(video_id: str) -> str:
  api = YouTubeTranscriptApi()
  transcript_list = api.fetch(video_id, languages=["en", "bn", "hi"])
  return " ".join(snippet.text for snippet in transcript_list)


def scrape_webpage_text(url: str, max_chars: int = 8000) -> str:
  headers = {"User-Agent": "Mozilla/5.0 (compatible; JARVIS-Bot/1.0)"}
  res = requests.get(url.strip(), headers=headers, timeout=15)
  res.raise_for_status()
  soup = BeautifulSoup(res.text, "html.parser")
  for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
    tag.decompose()
  lines = [
      line.strip()
      for line in soup.get_text(separator="\n").splitlines()
      if line.strip()
  ]
  return "\n".join(lines)[:max_chars]


# --- UNICODE / BENGALI PDF BUILDER ---
def find_unicode_font() -> str | None:
  candidates = [
      "fonts/NotoSansBengali-Regular.ttf",
      "fonts/HindSiliguri-Regular.ttf",
      "fonts/DejaVuSans.ttf",
      "NotoSansBengali-Regular.ttf",
      "DejaVuSans.ttf",
      "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  ]
  for path in candidates:
    if os.path.exists(path):
      return path
  return None


def build_story_pdf_bytes(story_text: str) -> bytes:
  pdf = FPDF()
  pdf.add_page()
  pdf.set_auto_page_break(auto=True, margin=15)

  font_path = find_unicode_font()
  if font_path:
    pdf.add_font("CustomUnicode", "", font_path)
    pdf.set_font("CustomUnicode", size=12)
    pdf.multi_cell(0, 8, story_text)
  else:
    pdf.set_font("Helvetica", size=12)
    safe_text = story_text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 8, safe_text)

  out = pdf.output(dest="S")
  if isinstance(out, (bytes, bytearray)):
    return bytes(out)
  return out.encode("latin-1", "replace")


# --- VIDEO SLIDESHOW EXPORTER ---
def render_slideshow_mp4(uploaded_files, seconds_per_slide=2.5, fps=24):
  if not uploaded_files:
    return None

  target_w, target_h = 1280, 720
  out_path = "jarvis_slideshow.mp4"
  writer = imageio.get_writer(
      out_path,
      format="FFMPEG",
      mode="I",
      fps=fps,
      codec="libx264",
      pixelformat="yuv420p",
  )
  try:
    frames_per_slide = max(1, int(round(seconds_per_slide * fps)))
    for up_file in uploaded_files:
      img = Image.open(up_file).convert("RGB")
      img_ratio = img.width / img.height
      target_ratio = target_w / target_h

      if img_ratio > target_ratio:
        new_w = target_w
        new_h = max(1, int(target_w / img_ratio))
      else:
        new_h = target_h
        new_w = max(1, int(target_h * img_ratio))

      resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
      canvas = Image.new("RGB", (target_w, target_h), (14, 17, 23))
      offset_x = (target_w - new_w) // 2
      offset_y = (target_h - new_h) // 2
      canvas.paste(resized, (offset_x, offset_y))

      frame = np.asarray(canvas, dtype=np.uint8)
      for _ in range(frames_per_slide):
        writer.append_data(frame)
  finally:
    writer.close()

  with open(out_path, "rb") as f:
    return f.read()


# --- MULTI-API ROUTER (4 FREE PROVIDERS + MEMORY) ---
def build_messages_payload(prompt: str, history: list | None = None):
  msgs = [{
      "role": "system",
      "content": (
          "You are JARVIS, an intelligent, concise personal AI assistant. "
          "Reply in the same language the user uses (Bengali or English)."
      ),
  }]
  if history:
    for m in history[-10:]:
      role = m.get("role")
      content = m.get("content", "")
      if role in ("user", "assistant") and content:
        msgs.append({"role": role, "content": content})
  if msgs[-1]["content"] != prompt:
    msgs.append({"role": "user", "content": prompt})
  return msgs


def build_gemini_contents(prompt: str, history: list | None = None):
  contents = []
  if history:
    for m in history[-10:]:
      role = "model" if m.get("role") == "assistant" else "user"
      text = m.get("content", "")
      if text:
        contents.append({"role": role, "parts": [{"text": text}]})
  if not contents or contents[-1]["parts"][0]["text"] != prompt:
    contents.append({"role": "user", "parts": [{"text": prompt}]})
  return contents


def call_ai_with_fallback(
    prompt,
    openrouter_keys,
    groq_keys,
    cerebras_keys,
    gemini_keys,
    history=None,
):
  error_log = []
  openai_msgs = build_messages_payload(prompt, history)
  gemini_contents = build_gemini_contents(prompt, history)

  # 1. OpenRouter Free Models
  if openrouter_keys:
    keys = [k.strip() for k in openrouter_keys.splitlines() if k.strip()]
    models = [
        "openrouter/free",
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
    ]
    for key in keys:
      for model in models:
        try:
          res = requests.post(
              url="https://openrouter.ai/api/v1/chat/completions",
              headers={
                  "Authorization": f"Bearer {key}",
                  "HTTP-Referer": "https://streamlit.io",
                  "X-Title": "JARVIS",
                  "Content-Type": "application/json",
              },
              json={"model": model, "messages": openai_msgs},
              timeout=30,
          )
          if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            return content, f"OpenRouter ({model.split('/')[-1]})"
          error_log.append(f"OR:{res.status_code}")
        except Exception as e:
          error_log.append(f"OR:{type(e).__name__}")

  # 2. Groq Free Tier
  if groq_keys:
    g_keys = [k.strip() for k in groq_keys.splitlines() if k.strip()]
    for gk in g_keys:
      for g_model in [
          "llama-3.3-70b-versatile",
          "llama-3.1-8b-instant",
          "gemma2-9b-it",
      ]:
        try:
          client = Groq(api_key=gk)
          completion = client.chat.completions.create(
              messages=openai_msgs,
              model=g_model,
          )
          return completion.choices[0].message.content, f"Groq ({g_model})"
        except Exception as e:
          error_log.append(f"GQ:{type(e).__name__}")

  # 3. Cerebras Free Tier
  if cerebras_keys:
    c_keys = [k.strip() for k in cerebras_keys.splitlines() if k.strip()]
    for ck in c_keys:
      for c_model in ["llama3.1-8b", "qwen-3-32b"]:
        try:
          res = requests.post(
              url="https://api.cerebras.ai/v1/chat/completions",
              headers={
                  "Authorization": f"Bearer {ck}",
                  "Content-Type": "application/json",
              },
              json={"model": c_model, "messages": openai_msgs},
              timeout=25,
          )
          if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            return content, f"Cerebras ({c_model})"
          error_log.append(f"CB:{res.status_code}")
        except Exception as e:
          error_log.append(f"CB:{type(e).__name__}")

  # 4. Google Gemini Free Tier
  if gemini_keys:
    gem_keys = [k.strip() for k in gemini_keys.splitlines() if k.strip()]
    for gem_k in gem_keys:
      try:
        client = genai.Client(api_key=gem_k)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_contents,
        )
        if response.text:
          return response.text, "Google Gemini 2.5 Flash"
      except Exception as e:
        error_log.append(f"GM:{type(e).__name__}")

  return (
      f"স্যার, সমস্ত ফ্রি এআই সার্ভার ব্যস্ত বা লিমিট শেষ। Errors: {error_log[:5]}",
      "Offline",
  )


# --- SIDEBAR & SECURITY ---
with st.sidebar:
  st.title("⚙️ JARVIS Settings")
  password = st.text_input("Enter Password", type="password")

  default_or = st.session_state.get(
      "or_keys", st.secrets.get("OPENROUTER_KEYS", "")
  )
  default_gq = st.session_state.get("gq_keys", st.secrets.get("GROQ_KEYS", ""))
  default_cb = st.session_state.get(
      "cb_keys", st.secrets.get("CEREBRAS_KEYS", "")
  )
  default_gm = st.session_state.get(
      "gm_keys", st.secrets.get("GEMINI_KEYS", "")
  )

  st.subheader("Free API Keys (1 per line)")
  openrouter_keys = st.text_area(
      "OpenRouter Keys", value=default_or, height=65, placeholder="sk-or-v1-..."
  )
  groq_keys = st.text_area(
      "Groq Keys", value=default_gq, height=65, placeholder="gsk_..."
  )
  cerebras_keys = st.text_area(
      "Cerebras Keys", value=default_cb, height=65, placeholder="csk-..."
  )
  gemini_keys = st.text_area(
      "Gemini Keys", value=default_gm, height=65, placeholder="AIzaSy..."
  )

  if st.button("Save Keys"):
    st.session_state["or_keys"] = openrouter_keys
    st.session_state["gq_keys"] = groq_keys
    st.session_state["cb_keys"] = cerebras_keys
    st.session_state["gm_keys"] = gemini_keys
    st.success("Keys saved for this session!")

  st.divider()
  tts_enabled = st.toggle("🔊 JARVIS Voice Reply (Free TTS)", value=False)
  tts_voice = st.selectbox(
      "Voice Accent",
      [
          "en-GB-RyanNeural",
          "en-US-GuyNeural",
          "bn-BD-PradeepNeural",
          "bn-BD-NabanitaNeural",
      ],
  )

expected_password = st.secrets.get("APP_PASSWORD", "jarvis2026")
if password != expected_password:
  st.warning("🔒 Access Denied. Please enter the correct password in the sidebar.")
  st.stop()

# --- MAIN UI ---
st.title("🤖 JARVIS Ultimate AI Workspace")
st.write("Welcome back, Sir. All systems and persistent storage are online.")

tabs = st.tabs([
    "🏠 Dashboard",
    "💬 Super Chat",
    "🌐 Web & YouTube",
    "💼 Freelance",
    "💡 Idea Vault",
    "📈 Market",
    "📚 PDF & Docs",
    "🎬 Video & Tools",
    "💰 Finance",
])


def handle_chat_turn(user_prompt: str, use_history: bool = True):
  history_snapshot = list(st.session_state.messages) if use_history else None
  st.session_state.messages.append(
      {"role": "user", "content": user_prompt, "provider": ""}
  )
  save_chat_message("user", user_prompt)

  with st.chat_message("user"):
    st.markdown(user_prompt)

  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      response, provider = call_ai_with_fallback(
          user_prompt,
          openrouter_keys,
          groq_keys,
          cerebras_keys,
          gemini_keys,
          history=history_snapshot,
      )
      st.markdown(response)
      st.caption(f"Powered by: {provider}")
      if tts_enabled and provider != "Offline":
        audio_bytes = synthesize_voice(response, tts_voice)
        if audio_bytes:
          st.audio(audio_bytes, format="audio/mp3")

  st.session_state.messages.append(
      {"role": "assistant", "content": response, "provider": provider}
  )
  save_chat_message("assistant", response, provider)


# --- TAB 1: DASHBOARD ---
with tabs[0]:
  st.header("Daily Briefing (Live Free Feeds)")
  c1, c2, c3 = st.columns(3)
  with c1:
    st.subheader("💵 USD to BDT")
    try:
      st.metric("Current Rate", f"{fetch_usd_bdt():.2f} BDT")
    except Exception:
      st.write("Rate unavailable.")
  with c2:
    st.subheader("₿ Bitcoin (USD)")
    try:
      crypto = fetch_crypto_prices()
      st.metric("BTC Price", f"${crypto['bitcoin']['usd']:,.0f}")
    except Exception:
      st.write("BTC unavailable.")
  with c3:
    st.subheader("Ξ Ethereum (USD)")
    try:
      crypto = fetch_crypto_prices()
      st.metric("ETH Price", f"${crypto['ethereum']['usd']:,.0f}")
    except Exception:
      st.write("ETH unavailable.")

  st.subheader("📰 Latest Tech & AI News")
  try:
    for r in fetch_tech_news(4):
      st.write(f"• **[{r['title']}]({r['url']})**")
  except Exception:
    st.write("News unavailable right now.")

# --- TAB 2: SUPER CHAT (PERSISTENT SQLITE HISTORY + VOICE) ---
with tabs[1]:
  st.header("Chat with JARVIS")
  if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history(40)

  top_c1, top_c2 = st.columns([1, 3])
  with top_c1:
    if st.button("Clear Chat"):
      clear_chat_history()
      st.session_state.messages = []
      st.rerun()
  with top_c2:
    memory_enabled = st.toggle("Use Conversation Memory", value=True)

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])
      if message.get("provider"):
        st.caption(f"Powered by: {message['provider']}")

  st.subheader("🎤 Voice Input (Free Groq Whisper)")
  audio = mic_recorder(
      start_prompt="Start Recording",
      stop_prompt="Stop Recording",
      just_once=True,
      key="jarvis_mic",
  )

  if audio and audio.get("bytes"):
    audio_hash = hash(audio["bytes"])
    if st.session_state.get("last_audio_hash") != audio_hash:
      st.session_state["last_audio_hash"] = audio_hash
      temp_path = "temp_audio.wav"
      try:
        with st.spinner("Transcribing voice..."):
          with open(temp_path, "wb") as f:
            f.write(audio["bytes"])

          g_list = [k.strip() for k in groq_keys.splitlines() if k.strip()]
          if g_list:
            client = Groq(api_key=g_list[0])
            with open(temp_path, "rb") as audio_file:
              transcription = client.audio.transcriptions.create(
                  file=(temp_path, audio_file.read()),
                  model="whisper-large-v3-turbo",
              )
            voice_prompt = transcription.text.strip()
            if voice_prompt:
              st.success(f"Transcribed: {voice_prompt}")
              handle_chat_turn(voice_prompt, use_history=memory_enabled)
          else:
            st.error("Free Groq API Key needed for Voice Input.")
      except Exception as e:
        st.error(f"Voice Error: {e}")
      finally:
        if os.path.exists(temp_path):
          os.remove(temp_path)

  if text_prompt := st.chat_input("Ask JARVIS anything..."):
    handle_chat_turn(text_prompt, use_history=memory_enabled)

# --- TAB 3: WEB & YOUTUBE SUMMARIZER ---
with tabs[2]:
  st.header("🌐 Free YouTube & Article Summarizer")
  sub_yt, sub_web = st.tabs(["📺 YouTube Summarizer", "📰 Web Article Reader"])

  with sub_yt:
    yt_url = st.text_input(
        "Paste YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=..."
    )
    yt_lang = st.selectbox("Summary Language", ["Bengali (বাংলা)", "English"])
    if st.button("Summarize YouTube Video"):
      vid_id = extract_youtube_id(yt_url)
      if not vid_id:
        st.warning("Valid YouTube link দিন স্যার।")
      else:
        try:
          with st.spinner("Fetching free captions & summarizing..."):
            transcript_text = get_youtube_transcript(vid_id)[:12000]
            prompt = (
                f"Summarize this YouTube video transcript in {yt_lang} with bullet points "
                f"and key takeaways:\n\n{transcript_text}"
            )
            summary, provider = call_ai_with_fallback(
                prompt, openrouter_keys, groq_keys, cerebras_keys, gemini_keys
            )
            st.caption(f"Powered by: {provider}")
            st.markdown(summary)
        except Exception as e:
          st.error(f"Could not fetch captions: {e}")

  with sub_web:
    web_url = st.text_input(
        "Paste Website / Blog URL:", placeholder="https://example.com/article"
    )
    web_question = st.text_input(
        "Custom instruction (optional):",
        value="Summarize the main points clearly.",
    )
    if st.button("Read & Summarize Webpage"):
      if not web_url.strip():
        st.warning("Please paste a valid URL.")
      else:
        try:
          with st.spinner("Reading webpage..."):
            page_text = scrape_webpage_text(web_url)
            prompt = f"{web_question}\n\nWebpage content:\n{page_text}"
            res, provider = call_ai_with_fallback(
                prompt, openrouter_keys, groq_keys, cerebras_keys, gemini_keys
            )
            st.caption(f"Powered by: {provider}")
            st.markdown(res)
        except Exception as e:
          st.error(f"Web Reader Error: {e}")

# --- TAB 4: FREELANCE ---
with tabs[3]:
  st.header("Freelance Proposal Generator")
  skills = st.text_input("Your Skills (e.g., Python, Graphic Design)")
  job_desc = st.text_area("Paste Job Description:")
  if st.button("Generate Proposal"):
    if not job_desc.strip():
      st.warning("Please paste a job description first.")
    else:
      prompt = (
          f"Write a concise, winning Upwork proposal for a freelancer skilled in "
          f"{skills or 'general tech'}. Job:\n{job_desc}"
      )
      res, provider = call_ai_with_fallback(
          prompt, openrouter_keys, groq_keys, cerebras_keys, gemini_keys
      )
      st.caption(f"Powered by: {provider}")
      st.text_area("Your Proposal:", res, height=220)

# --- TAB 5: IDEA VAULT ---
with tabs[4]:
  st.header("💡 Idea Vault")
  new_idea = st.text_input("Enter a new business or story idea:")
  if st.button("Save Idea"):
    if new_idea.strip():
      conn = get_conn()
      try:
        conn.execute(
            "INSERT INTO idea_vault (idea, created_at) VALUES (?, ?)",
            (new_idea.strip(), datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
        conn.commit()
        st.success("Idea saved!")
      finally:
        conn.close()
    else:
      st.warning("Please enter an idea before saving.")

  conn = get_conn()
  try:
    df_ideas = pd.read_sql_query(
        "SELECT * FROM idea_vault ORDER BY id DESC", conn
    )
  finally:
    conn.close()

  st.dataframe(df_ideas, use_container_width=True)

  if not df_ideas.empty:
    del_col1, del_col2, del_col3 = st.columns([2, 1, 2])
    with del_col1:
      del_id = st.selectbox("Select Idea ID to Delete", df_ideas["id"].tolist())
    with del_col2:
      st.write("")
      if st.button("Delete Idea"):
        conn = get_conn()
        try:
          conn.execute("DELETE FROM idea_vault WHERE id = ?", (int(del_id),))
          conn.commit()
          st.rerun()
        finally:
          conn.close()
    with del_col3:
      st.write("")
      st.download_button(
          "Export Ideas CSV",
          data=df_ideas.to_csv(index=False).encode("utf-8"),
          file_name="jarvis_ideas.csv",
          mime="text/csv",
      )

# --- TAB 6: MARKET ---
with tabs[5]:
  st.header("📈 Market Analyzer")
  asset = st.text_input("Asset Name (e.g., Bitcoin, Apple):")
  if st.button("Analyze Risk"):
    if not asset.strip():
      st.warning("Please enter an asset name.")
    else:
      prompt = (
          f"Provide a structured risk vs. reward overview for {asset.strip()}. "
          "Include key risks and a clear reminder that this is not financial advice."
      )
      res, provider = call_ai_with_fallback(
          prompt, openrouter_keys, groq_keys, cerebras_keys, gemini_keys
      )
      st.caption(f"Powered by: {provider}")
      st.markdown(res)

# --- TAB 7: PDF & DOCS ---
with tabs[6]:
  st.header("📚 AI Novelist, PDF Reader & Exporter")
  doc_tab1, doc_tab2 = st.tabs(["✍️ Story & PDF Writer", "📄 Chat with Uploaded PDF"])

  with doc_tab1:
    novel_prompt = st.text_area("Story or article idea:")
    if st.button("Write & Export"):
      if not novel_prompt.strip():
        st.warning("Please enter a story idea.")
      else:
        with st.spinner("Writing..."):
          story, provider = call_ai_with_fallback(
              f"Write a creative, engaging piece based on:\n{novel_prompt.strip()}",
              openrouter_keys,
              groq_keys,
              cerebras_keys,
              gemini_keys,
          )
          st.session_state["last_story"] = story
          st.session_state["last_story_provider"] = provider

    if st.session_state.get("last_story"):
      story_out = st.session_state["last_story"]
      st.caption(
          f"Powered by: {st.session_state.get('last_story_provider', '')}"
      )
      st.markdown(story_out)

      d_col1, d_col2 = st.columns(2)
      with d_col1:
        try:
          pdf_bytes = build_story_pdf_bytes(story_out)
          st.download_button(
              "Download PDF",
              data=pdf_bytes,
              file_name="JARVIS_Document.pdf",
              mime="application/pdf",
          )
        except Exception as e:
          st.error(f"PDF Error: {e}")
      with d_col2:
        st.download_button(
            "Download Markdown (.md)",
            data=story_out.encode("utf-8"),
            file_name="JARVIS_Document.md",
            mime="text/markdown",
        )

  with doc_tab2:
    uploaded_pdf = st.file_uploader("Upload any PDF file", type=["pdf"])
    pdf_q = st.text_input(
        "What do you want to know from this PDF?",
        value="Summarize the key points of this document.",
    )
    if st.button("Analyze PDF"):
      if not uploaded_pdf:
        st.warning("Please upload a PDF first.")
      else:
        try:
          with st.spinner("Reading PDF..."):
            reader = PdfReader(uploaded_pdf)
            extracted = "\n".join(
                page.extract_text() or "" for page in reader.pages[:15]
            )[:12000]
            if not extracted.strip():
              st.warning("No selectable text found in this PDF.")
            else:
              prompt = f"{pdf_q}\n\nPDF Text:\n{extracted}"
              ans, provider = call_ai_with_fallback(
                  prompt, openrouter_keys, groq_keys, cerebras_keys, gemini_keys
              )
              st.caption(f"Powered by: {provider}")
              st.markdown(ans)
        except Exception as e:
          st.error(f"PDF Read Error: {e}")

# --- TAB 8: VIDEO & FREE TOOLS ---
with tabs[7]:
  st.header("🎬 Video Slideshow & QR Studio")
  v_col1, v_col2 = st.columns(2)

  with v_col1:
    st.subheader("🎞️ Image-to-MP4 Slideshow")
    uploaded_images = st.file_uploader(
        "Upload Scene Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    slide_sec = st.slider(
        "Seconds per image", min_value=1.0, max_value=8.0, value=2.5, step=0.5
    )
    if st.button("Render MP4 Video"):
      if not uploaded_images:
        st.warning("Upload images first.")
      else:
        try:
          with st.spinner("Rendering MP4..."):
            mp4_bytes = render_slideshow_mp4(
                uploaded_images, seconds_per_slide=slide_sec
            )
          if mp4_bytes:
            st.video(mp4_bytes)
            st.download_button(
                "Download MP4",
                data=mp4_bytes,
                file_name="JARVIS_Slideshow.mp4",
                mime="video/mp4",
            )
        except Exception as e:
          st.error(f"Video error: {e}")

  with v_col2:
    st.subheader("📱 Free QR Code Generator")
    qr_data = st.text_input(
        "Enter URL or Text for QR Code:", placeholder="https://..."
    )
    if st.button("Generate QR Code"):
      if not qr_data.strip():
        st.warning("Enter text or URL first.")
      else:
        qr_img = qrcode.make(qr_data.strip())
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        qr_bytes = buf.getvalue()
        st.image(qr_bytes, width=220)
        st.download_button(
            "Download QR PNG",
            data=qr_bytes,
            file_name="jarvis_qr.png",
            mime="image/png",
        )

# --- TAB 9: FINANCE ---
with tabs[8]:
  st.header("💰 Finance Tracker")
  fin_type = st.selectbox("Type", ["Income", "Expense"])
  amount = st.number_input("Amount (BDT)", min_value=1.0, step=10.0)
  category = st.text_input("Category", placeholder="e.g., Freelance, Food, Hosting")
  if st.button("Add Transaction"):
    if not category.strip():
      st.warning("Please enter a category.")
    else:
      conn = get_conn()
      try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO finance (type, amount, category, date) VALUES (?, ?, ?, ?)",
            (fin_type, float(amount), category.strip(), date_str),
        )
        conn.commit()
        st.success("Transaction saved!")
      finally:
        conn.close()

  conn = get_conn()
  try:
    df_fin = pd.read_sql_query("SELECT * FROM finance ORDER BY id DESC", conn)
  finally:
    conn.close()

  if not df_fin.empty:
    total_income = df_fin[df_fin["type"] == "Income"]["amount"].sum()
    total_expense = df_fin[df_fin["type"] == "Expense"]["amount"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Income", f"{total_income:,.2f} BDT")
    c2.metric("Total Expense", f"{total_expense:,.2f} BDT")
    c3.metric("Net Balance", f"{total_income - total_expense:,.2f} BDT")

    exp_df = df_fin[df_fin["type"] == "Expense"]
    if not exp_df.empty:
      st.subheader("Expenses by Category")
      st.bar_chart(exp_df.groupby("category")["amount"].sum())

  st.dataframe(df_fin, use_container_width=True)

  if not df_fin.empty:
    f_del1, f_del2, f_del3 = st.columns([2, 1, 2])
    with f_del1:
      del_fin_id = st.selectbox(
          "Transaction ID to Delete", df_fin["id"].tolist()
      )
    with f_del2:
      st.write("")
      if st.button("Delete Entry"):
        conn = get_conn()
        try:
          conn.execute("DELETE FROM finance WHERE id = ?", (int(del_fin_id),))
          conn.commit()
          st.rerun()
        finally:
          conn.close()
    with f_del3:
      st.write("")
      st.download_button(
          "Export Finance CSV",
          data=df_fin.to_csv(index=False).encode("utf-8"),
          file_name="jarvis_finance.csv",
          mime="text/csv",
      )

  if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
      st.download_button(
          "Download Full DB Backup (.db)",
          data=f.read(),
          file_name="jarvis_backup.db",
          mime="application/octet-stream",
      )

