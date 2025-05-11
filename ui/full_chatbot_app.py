import streamlit as st
from datetime import datetime
import sqlite3
import pandas as pd
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.translator import translate_to_english, translate_from_english
from utils.patient_profiles import get_patient_profile
from scripts.rag_chatbot import load_llm, load_embeddings, SimpleVectorDB, custom_retriever
import matplotlib.pyplot as plt
import tempfile
from gtts import gTTS
import base64
import speech_recognition as sr
import soundfile as sf
import json
from collections import Counter

# --- Load Alignment Dataset ---
ALIGNMENT_DATA_PATH = "data/alignment_dataset.json"
if os.path.exists(ALIGNMENT_DATA_PATH):
    with open(ALIGNMENT_DATA_PATH, "r", encoding="utf-8") as f:
        alignment_data = json.load(f)
else:
    alignment_data = []

# --- Setup DB ---
conn = sqlite3.connect("chat_history.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS history (
    user_id TEXT,
    timestamp TEXT,
    language TEXT,
    question TEXT,
    context TEXT,
    answer TEXT,
    feedback TEXT
)''')
conn.commit()

# --- App Config ---
st.set_page_config(page_title="Multilingual Diabetes Chatbot", layout="wide")
st.title("🧑‍⚕️ Diabetes Management Assistant (Multilingual)")

# --- Sidebar ---
st.sidebar.header("User Profile")
user_id = st.sidebar.selectbox("Select Patient", ["elena", "miguel", "carmen"])
language = st.sidebar.selectbox("Preferred Language", ["English", "Spanish", "Swahili"])

profile = get_patient_profile(user_id)
if not profile:
    st.sidebar.error("No profile found for selected user.")
    st.stop()

# --- Session State Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load Models + VectorDB ---
@st.cache_resource
def setup_pipeline():
    embeddings = load_embeddings()
    vectordb = SimpleVectorDB.load("numpy_vectordb", embeddings)
    llm = load_llm()
    return vectordb, llm

vectordb, llm = setup_pipeline()

# --- Voice Input ---
def record_and_transcribe():
    st.write("🎤 Recording... (say something then stop)")
    audio_file = st.file_uploader("Upload your voice input (WAV only)", type=["wav"])
    if audio_file is not None:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language='sw' if language == "Swahili" else 'es' if language == "Spanish" else 'en')
                st.success(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
    return ""

# --- Input box or voice ---
st.markdown("**Choose input method:**")
input_mode = st.radio("Input Method", ["Text", "Voice"])
user_input = ""

if input_mode == "Text":
    user_input = st.text_input("Ask a diabetes-related question")
elif input_mode == "Voice":
    user_input = record_and_transcribe()

submit = st.button("Send")

if submit and user_input:
    ts = datetime.utcnow().isoformat()

    # Translate input if needed
    input_translated = translate_to_english(user_input, language) if language != "English" else user_input

    # Retrieve context with timing and score
    retrieval_start = time.time()
    docs = custom_retriever(vectordb, input_translated)
    retrieval_time = time.time() - retrieval_start
    context = "\n".join(docs)

    # Show retrieved source citations
    with st.expander("🔍 Retrieved Source Passages"):
        for i, doc in enumerate(docs[:3]):
            st.markdown(f"**Passage {i+1}:**")
            st.markdown(doc.strip()[:400] + ("..." if len(doc.strip()) > 400 else ""))

    # Generate answer with timing
    prompt = f"""
You are a medically accurate assistant for people with diabetes. Always follow ADA, WHO, or CDC guidelines when answering.

Patient Profile:
- Name: {profile['name']}
- Age: {profile['age']}
- Diagnoses: {', '.join(profile['diagnoses'])}
- Medications: {', '.join(profile['medications'])} (Adherence: {profile['adherence']})
- Cultural Beliefs: {profile['culture']}
- Social Background: {profile['social']}

Follow these guidelines:
1. Cross-check response validity with authoritative guidelines (ADA, WHO, CDC, NIH)
2. Label answers with ✅ if aligned with guidelines, or ⚠️ if advisory only.
3. Be honest about limitations.

Context:
{context}

Question: {input_translated}
Answer:"""

    llm_start = time.time()
    answer = llm(prompt)
    llm_time = time.time() - llm_start
    answer_translated = translate_from_english(answer, language) if language != "English" else answer

    # Display chat
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Bot:** {answer_translated.strip()}")

    # Text-to-speech
    tts = gTTS(text=answer_translated.strip(), lang='sw' if language == 'Swahili' else 'es' if language == 'Spanish' else 'en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        audio_bytes = open(f.name, 'rb').read()
        st.audio(audio_bytes, format='audio/mp3')

    # Save history
    st.session_state.chat_history.append((user_input, answer_translated))
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, '')",
              (user_id, ts, language, user_input, context[:2000], answer_translated.strip()))
    conn.commit()

    # Show performance metrics
    st.markdown("### ⏱️ Performance")
    st.markdown(f"- Retrieval time: `{retrieval_time:.2f}` seconds")
    st.markdown(f"- LLM time: `{llm_time:.2f}` seconds")
    fig, ax = plt.subplots()
    ax.bar(["Retrieval", "LLM"], [retrieval_time, llm_time], color=['blue', 'green'])
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Query Processing Time")
    st.pyplot(fig)

# --- Chat History Display ---
with st.expander("✉️ Chat History"):
    rows = c.execute("SELECT timestamp, question, answer, feedback FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,)).fetchall()
    for ts, q, a, fb in rows:
        st.markdown(f"**[{ts[:19]}] You:** {q}")
        st.markdown(f"**Bot:** {a}")
        if fb:
            st.markdown(f"_Feedback: {fb}_")

# --- Feedback Collection ---
st.markdown("---")
st.subheader("📈 Feedback")
if st.session_state.chat_history:
    latest_q, latest_a = st.session_state.chat_history[-1]
    feedback = st.radio("Was this answer helpful?", ["", "Yes", "No"])
    comments = st.text_area("Additional comments")
    if st.button("Submit Feedback") and feedback:
        c.execute("""
            UPDATE history SET feedback=?
            WHERE user_id=? AND question=? AND answer=?
        """, (f"{feedback} - {comments}", user_id, latest_q, latest_a))
        conn.commit()
        st.success("Thanks for your feedback!")

# --- Export CSV ---
st.markdown("---")
st.subheader("📁 Export Chat History")
if st.button("Download CSV"):
    export_data = c.execute("SELECT * FROM history WHERE user_id=?", (user_id,)).fetchall()
    df = pd.DataFrame(export_data, columns=["User ID", "Timestamp", "Language", "Question", "Context", "Answer", "Feedback"])
    csv = df.to_csv(index=False)
    st.download_button("Download as CSV", data=csv, file_name=f"chat_history_{user_id}.csv", mime="text/csv")

# --- Dashboard ---
st.markdown("---")
st.subheader("📊 Interactive Visualizations")
df = pd.read_sql_query("SELECT * FROM history", conn)

if not df.empty:
    with st.tabs(["Top Medications", "Query Frequency"])[0]:
        meds = []
        for u in ["elena", "miguel", "carmen"]:
            p = get_patient_profile(u)
            meds.extend(p["medications"])
        med_counts = Counter(meds)
        med_df = pd.DataFrame(med_counts.items(), columns=["Medication", "Count"])
        st.bar_chart(med_df.set_index("Medication"))

    with st.tabs(["Top Medications", "Query Frequency"])[1]:
        ts_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
        df["date"] = pd.to_datetime(df[ts_col]).dt.date
        freq = df.groupby("date")["question"].count()
        st.line_chart(freq)
