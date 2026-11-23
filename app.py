import streamlit as st
import requests
import sqlite3
import pandas as pd
import os
import time
import json
from datetime import datetime
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from fpdf import FPDF
from groq import Groq
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from streamlit_mic_recorder import mic_recorder

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="JARVIS Ultimate", page_icon="🤖", layout="wide")

# --- CUSTOM CSS THEME ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #00d2ff; }
    h1, h2, h3 { color: #00d2ff !important; }
    .stButton>button { background-color: #00d2ff; color: black; border-radius: 5px; font-weight: bold;}
    .stTextArea textarea { background-color: #1e1e1e; color: white; }
    .stTextInput input { background-color: #1e1e1e; color: white; }
</style>
""", unsafe_allow_html=True)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('jarvis.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS idea_vault (id INTEGER PRIMARY KEY, idea TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS finance (id INTEGER PRIMARY KEY, type TEXT, amount REAL, category TEXT, date TEXT)''')
    conn.commit()
    conn.close()
init_db()

# --- MULTI-API ROUTER (LATEST FREE MODELS) ---
def call_ai_with_fallback(prompt, openrouter_keys, groq_keys, gemini_keys):
    error_log = []
    
    # 1. Try OpenRouter (Auto Router & Latest Free Models)
    if openrouter_keys:
        keys = [k.strip() for k in openrouter_keys.split('\n') if k.strip()]
        models = [
            "openrouter/free", # Auto-router (সবচেয়ে ভালো ফ্রি মডেল বেছে নেবে)
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "microsoft/phi-3-mini-128k-instruct:free"
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
                            "Content-Type": "application/json"
                        },
                        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                        timeout=20
                    )
                    if res.status_code == 200:
                        return res.json()['choices'][0]['message']['content'], f"OpenRouter ({model.split('/')[-1]})"
                    else:
                        error_log.append(f"OR: {res.status_code}")
                except Exception as e:
                    error_log.append(f"OR: {str(e)[:20]}")

    # 2. Try Groq (Latest Models)
    if groq_keys:
        g_keys = [k.strip() for k in groq_keys.split('\n') if k.strip()]
        for gk in g_keys:
            try:
                client = Groq(api_key=gk)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant", # আপডেটেড মডেল
                )
                return chat_completion.choices[0].message.content, "Groq (Llama 3.1)"
            except Exception as e:
                error_log.append(f"GQ: {str(e)[:20]}")
            
            try:
                client = Groq(api_key=gk)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gemma2-9b-it", # ব্যাকআপ মডেল
                )
                return chat_completion.choices[0].message.content, "Groq (Gemma 2)"
            except Exception as e:
                error_log.append(f"GQ2: {str(e)[:20]}")

    # 3. Try Gemini (Latest Models)
    if gemini_keys:
        gem_keys = [k.strip() for k in gemini_keys.split('\n') if k.strip()]
        for gem_k in gem_keys:
            try:
                genai.configure(api_key=gem_k)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                return response.text, "Google Gemini 1.5 Flash"
            except Exception as e:
                error_log.append(f"GM: {str(e)[:20]}")
            
            try:
                genai.configure(api_key=gem_k)
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)
                return response.text, "Google Gemini 1.5 Pro"
            except Exception as e:
                error_log.append(f"GM2: {str(e)[:20]}")

    return f"স্যার, সমস্ত এআই সার্ভার ব্যস্ত বা লিমিট শেষ। Error Details: {str(error_log[:2])}", "Offline"

# --- SIDEBAR & SECURITY ---
with st.sidebar:
    st.title("⚙️ JARVIS Settings")
    password = st.text_input("Enter Password", type="password")
    
    st.subheader("API Keys (1 per line)")
    openrouter_keys = st.text_area("OpenRouter Keys", height=80, placeholder="Key 1\nKey 2\nKey 3")
    groq_keys = st.text_area("Groq Keys", height=80, placeholder="Key 1\nKey 2\nKey 3")
    gemini_keys = st.text_area("Gemini Keys", height=80, placeholder="Key 1\nKey 2\nKey 3")
    
    if st.button("Save Keys"):
        st.success("Keys configured!")

# App Password Check (Default is jarvis2026)
if password != st.secrets.get("APP_PASSWORD", "jarvis2026"):
    st.warning("🔒 Access Denied. Please enter the correct password in the sidebar.")
    st.stop()

# --- MAIN UI ---
st.title("🤖 JARVIS Ultimate AI Workspace")
st.write("Welcome back, Sir. All systems are operational.")

tabs = st.tabs(["🏠 Dashboard", "💬 Super Chat", "💼 Freelance", "💡 Idea Vault", "📈 Market", "📚 Documents", "🎬 Video", "💰 Finance"])

# --- TAB 1: DASHBOARD ---
with tabs[0]:
    st.header("Daily Briefing")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💵 USD to BDT (Approx)")
        try:
            res = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
            rate = res.json()['rates']['BDT']
            st.metric("Current Rate", f"{rate:.2f} BDT")
        except:
            st.write("Data unavailable right now.")
    with col2:
        st.subheader("📰 Latest Tech News")
        try:
            results = DDGS().news(keywords="technology", max_results=3)
            for r in results:
                st.write(f"**[{r['title']}]({r['url']})**")
        except:
            st.write("News unavailable.")

# --- TAB 2: SUPER CHAT (Text + Voice) ---
with tabs[1]:
    st.header("Chat with JARVIS")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Voice Input Module
    st.subheader("🎤 Voice Input (Groq Whisper)")
    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", just_once=True)
    
    if audio:
        with st.spinner("Transcribing voice..."):
            try:
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio['bytes'])
                
                if groq_keys:
                    gk = [k.strip() for k in groq_keys.split('\n') if k.strip()][0]
                    client = Groq(api_key=gk)
                    with open("temp_audio.wav", "rb") as file:
                        transcription = client.audio.transcriptions.create(
                            file=(file.name, file.read()),
                            model="whisper-large-v3-turbo",
                        )
                    prompt = transcription.text
                    st.success(f"Transcribed: {prompt}")
                    
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response, provider = call_ai_with_fallback(prompt, openrouter_keys, groq_keys, gemini_keys)
                            st.markdown(response)
                            st.caption(f"Powered by: {provider}")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Groq API Key needed for Voice Input.")
                
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")
            except Exception as e:
                st.error(f"Voice Error: {e}")

    # Text Chat Input
    if prompt := st.chat_input("Ask JARVIS anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, provider = call_ai_with_fallback(prompt, openrouter_keys, groq_keys, gemini_keys)
                st.markdown(response)
                st.caption(f"Powered by: {provider}")
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- TAB 3: FREELANCE ---
with tabs[2]:
    st.header("Freelance Proposal Generator")
    skills = st.text_input("Your Skills (e.g., Python, Graphic Design)")
    job_desc = st.text_area("Paste Job Description:")
    if st.button("Generate Proposal"):
        prompt = f"Write a winning Upwork proposal for a freelancer skilled in {skills}. Job: {job_desc}"
        res, _ = call_ai_with_fallback(prompt, openrouter_keys, groq_keys, gemini_keys)
        st.text_area("Your Proposal:", res, height=200)

# --- TAB 4: IDEA VAULT ---
with tabs[3]:
    st.header("💡 Idea Vault")
    new_idea = st.text_input("Enter a new business or story idea:")
    if st.button("Save Idea"):
        conn = sqlite3.connect('jarvis.db')
        c = conn.cursor()
        c.execute("INSERT INTO idea_vault (idea) VALUES (?)", (new_idea,))
        conn.commit()
        conn.close()
        st.success("Idea saved!")
    
    conn = sqlite3.connect('jarvis.db')
    df = pd.read_sql_query("SELECT * FROM idea_vault", conn)
    conn.close()
    st.dataframe(df)

# --- TAB 5: MARKET ---
with tabs[4]:
    st.header("📈 Market Analyzer")
    asset = st.text_input("Asset Name (e.g., Bitcoin, Apple):")
    if st.button("Analyze Risk"):
        prompt = f"Analyze risk vs reward for {asset}. Warn against high risk."
        res, _ = call_ai_with_fallback(prompt, openrouter_keys, groq_keys, gemini_keys)
        st.write(res)

# --- TAB 6: DOCUMENTS ---
with tabs[5]:
    st.header("📚 AI Novelist & PDF")
    novel_prompt = st.text_area("Story idea:")
    if st.button("Write & Download PDF"):
        with st.spinner("Writing story..."):
            story, _ = call_ai_with_fallback(f"Write a short story: {novel_prompt}", openrouter_keys, groq_keys, gemini_keys)
            st.write(story)
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, story.encode('latin-1', 'replace').decode('latin-1'))
                pdf.output("story.pdf")
                with open("story.pdf", "rb") as f:
                    st.download_button("Download PDF", f, file_name="JARVIS_Story.pdf")
            except Exception as e:
                st.error(f"PDF Error: {e}")

# --- TAB 7: VIDEO ---
with tabs[6]:
    st.header("🎬 Video Studio")
    st.warning("স্যার, ফ্রি সার্ভারে ভিডিও রেন্ডার করতে অনেক সময় লাগে বা ক্র্যাশ করে। তাই এটি Safe Mode-এ আছে।")
    uploaded_images = st.file_uploader("Upload Scene Images", accept_multiple_files=True)
    if st.button("Render Video"):
        if uploaded_images:
            st.success("Images received! (Video rendering module is in safe mode).")
        else:
            st.warning("Please upload images first.")

# --- TAB 8: FINANCE ---
with tabs[7]:
    st.header("💰 Finance Tracker")
    fin_type = st.selectbox("Type", ["Income", "Expense"])
    amount = st.number_input("Amount (BDT)", min_value=1.0)
    category = st.text_input("Category")
    if st.button("Add Transaction"):
        conn = sqlite3.connect('jarvis.db')
        c = conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d")
        c.execute("INSERT INTO finance (type, amount, category, date) VALUES (?, ?, ?, ?)", (fin_type, amount, category, date))
        conn.commit()
        conn.close()
        st.success("Transaction saved!")
    
    conn = sqlite3.connect('jarvis.db')
    df_fin = pd.read_sql_query("SELECT * FROM finance", conn)
    conn.close()
    st.dataframe(df_fin)
    
    with open("jarvis.db", "rb") as f:
        st.download_button("Download Backup", f, file_name="jarvis_backup.db")
