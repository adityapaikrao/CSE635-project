import streamlit as st
import rag_chatbot as rag_chatbot
from simple_vector_search import SimpleVectorDB

# Streamlit Page Config
st.set_page_config(layout="wide", page_title="Diabetes RAG Chatbot", page_icon="🧠")

st.sidebar.header("Diabetes Assistant 🤖")
st.sidebar.write("A Retrieval-Augmented Generation chatbot for diabetes-related queries.")
st.sidebar.write("Provide any existing conditions to personalize responses.")

# Patient condition input (persistent)
patient_conditions = st.sidebar.text_input("Existing Conditions (optional)", "")

# Main window
st.title("💬 Diabetes Chatbot")

# Load backend resources
@st.cache_resource
def load_resources():
    embeddings = rag_chatbot.load_embeddings()
    vectordb = SimpleVectorDB.load(rag_chatbot.DB_PATH, embeddings)
    llm = rag_chatbot.load_llm()
    return vectordb, llm

vectordb, llm = load_resources()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Show chat history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User query
if query := st.chat_input("Ask your diabetes-related question..."):

    # Add user message
    st.session_state['messages'].append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve context
    retrieved_docs = rag_chatbot.custom_retriever(vectordb, query)

    # Language detection
    language_hint = "ES" if any(word in query.lower() for word in ["el", "la", "los", "las", "qué", "cómo", "para", "dónde", "por"]) else "EN"

    # Build prompt
    prompt = f"""You are a medically‑accurate diabetes assistant.
Respond in the same language as the question ({'Spanish' if language_hint=='ES' else 'English'}).

PATIENT CONDITIONS: {patient_conditions}
RETRIEVED INFO:
{retrieved_docs}

QUESTION: {query}

ANSWER:"""

    # Call the local LLM
    try:
        response_text = llm(prompt)
        answer = rag_chatbot.post_process_response(response_text)
    except Exception as e:
        answer = f"Error generating response: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    # Add assistant response to history
    st.session_state['messages'].append({"role": "assistant", "content": answer})
