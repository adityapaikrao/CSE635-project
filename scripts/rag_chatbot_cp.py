#!/usr/bin/env python
# scripts/rag_chatbot.py

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from simple_vector_search import SimpleVectorDB
import torch
import logging
import time
import os
import psutil
from peft import PeftModel
from datetime import datetime
import sys
import streamlit as st

# --- Logging Setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"rag_chatbot_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_chatbot")

# --- Config ---
MODEL_NAME = "bigscience/bloomz-1b7"  # More efficient model
LORA_WEIGHTS_PATH = "./lora-bloomz-diabetes"
DB_PATH = "numpy_vectordb"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def filter_docs_by_score(docs, min_score=0.60):
    """Filter out low-confidence retrievals based on cosine similarity score."""
    filtered_docs = [doc for doc in docs if doc.get("score", 0) >= min_score]
    # If no documents meet the threshold, return at least the best one available
    if not filtered_docs and docs:
        return [max(docs, key=lambda x: x.get("score", 0))]
    return filtered_docs

def log_system_info():
    """Log system information for debugging purposes."""
    logger.info("System information:")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        logger.info(f"GPU available! Found {device_count} GPU(s): {device_names}")
        for i in range(device_count):
            memory_info = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {device_names[i]} with {memory_info:.2f} GB memory")
    
    logger.info(f"CPU count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    memory = psutil.virtual_memory()
    logger.info(f"System memory: {memory.total / 1024**3:.2f} GB total, {memory.available / 1024**3:.2f} GB available")

# --- Load Embeddings ---
def load_embeddings():
    logger.info(f"Loading embeddings model: {EMBEDDING_MODEL}")
    start_time = time.time()
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    end_time = time.time()
    logger.info(f"Embeddings model loaded in {end_time - start_time:.2f} seconds")
    
    return embeddings

# --- Custom Text Generation Function For PeftModel ---
def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, repetition_penalty=1.1, do_sample=True):
    """Custom text generation function that works with PeftModel."""
    logger.info("Using custom text generation function for PeftModel")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Get only the newly generated text (remove the prompt)
    response = generated_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    
    return response

# --- Load LLM ---
def load_llm():
    logger.info(f"Starting to load LLM model: {MODEL_NAME}")
    start_time = time.time()
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info("Configuring 4-bit quantization for memory efficiency...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        logger.info("Loading base model onto device...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config
        )
        
        # Create a wrapper class for PeftModel
        class PeftLLMWrapper:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
            
            def __call__(self, prompt):
                return generate_text(self.model, self.tokenizer, prompt)
        
        if LORA_WEIGHTS_PATH:
            logger.info(f"Applying LoRA weights from: {LORA_WEIGHTS_PATH}")
            model = PeftModel.from_pretrained(model, LORA_WEIGHTS_PATH, is_trainable=False)
            logger.info("Successfully applied LoRA weights")
            
            # Return the wrapper instead of using the pipeline
            logger.info("Creating custom LLM wrapper for PeftModel")
            llm = PeftLLMWrapper(model, tokenizer)
            return llm
        
        # If no LoRA weights, use the standard pipeline approach
        logger.info("Creating text generation pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.info("Falling back to TinyLlama...")

        fallback_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            device_map="auto",
            torch_dtype=torch.float16
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            repetition_penalty=1.1
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

    finally:
        end_time = time.time()
        logger.info(f"LLM loading process completed in {end_time - start_time:.2f} seconds")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
            logger.info(f"GPU memory reserved: {memory_reserved:.2f} GB")

# --- Custom Retriever ---
def custom_retriever(vectordb, query):
    """Enhanced custom retriever with better relevance filtering."""
    logger.info(f"Processing query: '{query}'")
    start_time = time.time()

    # Try with a larger k to get more potential matches
    results = vectordb.similarity_search(query, k=8)
    
    # Add scores to log for diagnostic purposes
    score_log = [f"{i+1}: {r.get('score', 0):.4f}" for i, r in enumerate(results[:5])] if results else []
    if score_log:
        logger.info(f"Top retrieval scores: {', '.join(score_log)}")
    
    # Apply stricter filtering
    filtered_results = filter_docs_by_score(results, min_score=0.60)

    end_time = time.time()
    retrieval_time = end_time - start_time
    logger.info(f"Retrieved {len(results)} raw docs, {len(filtered_results)} after filtering in {retrieval_time:.4f} seconds")

    # Extract text from filtered results
    if filtered_results:
        if isinstance(filtered_results[0], dict) and 'text' in filtered_results[0]:
            return [r['text'] for r in filtered_results]
        elif hasattr(filtered_results[0], "page_content"):
            return [r.page_content for r in filtered_results]
        elif hasattr(filtered_results[0], "text"):
            return [r.text for r in filtered_results]
        else:
            return [str(r) for r in filtered_results]
    
    # Add fallback information for common diabetes topics
    # This ensures there's always some relevant context
    medical_facts = {
        "a1c": "A1C (glycated hemoglobin) is a blood test that measures your average blood sugar levels over the past 2-3 months. For most adults with diabetes, the target A1C is less than 7%. The test works by measuring the percentage of hemoglobin proteins in your blood that are coated with sugar (glycated). The higher your A1C level, the higher your risk of diabetes complications.",
        "foot": "Diabetes can cause nerve damage (neuropathy) and poor circulation in your feet. This can lead to numbness, tingling, pain, and wounds that heal slowly. It's important to check your feet daily for cuts, blisters, redness, or swelling. If you notice any foot problems, contact your healthcare provider promptly.",
        "diet": "A healthy diet for diabetes focuses on controlling carbohydrate intake, choosing high-fiber foods, limiting refined carbs and sugars, and emphasizing vegetables, whole grains, lean proteins, and healthy fats. Portion control is also important.",
        "exercise": "Regular physical activity helps control blood sugar levels by improving your body's insulin sensitivity. Aim for 150 minutes of moderate-intensity exercise per week. Always consult your healthcare provider before starting a new exercise program."
    }
    
    # Determine if we should add fallback information
    query_lower = query.lower()
    additional_context = []
    
    for key, info in medical_facts.items():
        if key in query_lower:
            additional_context.append(f"MEDICAL FACT: {info}")
    
    if additional_context:
        logger.info(f"Adding {len(additional_context)} fallback medical facts to context")
        return additional_context
    
    return []

def build_prompt(query, context, patient_conditions, language_hint="EN"):
    prompt = f"""You are a medically accurate assistant for people with diabetes. Always prioritize factual, evidence-based information.
This user has the following conditions: {patient_conditions}. Answer in the same language as the question ({'Spanish' if language_hint == 'ES' else 'English'}).

Follow these guidelines:
1. If the retrieved information seems incorrect or contradictory, rely on established medical facts
2. If you're unsure about something, be honest about limitations rather than providing potentially incorrect advice
3. For questions about medical conditions, recommend consulting a healthcare provider when appropriate
4. Provide complete, coherent answers without trailing off mid-explanation
5. When addressing nutrition, focus on foods that help maintain steady blood sugar levels

Use the context below to answer their question. If the context doesn't contain relevant information, rely on general diabetes knowledge.

Context:
{context}

Question: {query}
Answer:"""
    print(f'prompt: {prompt}')
    return prompt

# --- Post-processing helper ---
def post_process_response(text):
    return text.strip().split("ANSWER:")[-1].strip()


# --- Streamlit-based RAG chatbot ---
# --- Streamlit-based RAG chatbot ---
def run_chatbot():
    st.set_page_config(layout="wide", page_title="Diabetes RAG Chatbot", page_icon="🧠")

    logger.info("Initializing Streamlit RAG Chatbot UI...")

    st.sidebar.header("Diabetes Assistant 🤖")
    st.sidebar.write("A Retrieval-Augmented Generation chatbot for diabetes-related queries.")
    st.sidebar.write("Provide any existing conditions to personalize responses.")
    patient_conditions = st.sidebar.text_input("Existing Conditions (optional)", "")

    st.title("💬 Diabetes Chatbot")

    # Load vector DB, embeddings, and LLM (only once)
    @st.cache_resource
    def load_resources():
        logger.info("Loading embeddings, vector DB, and LLM...")
        embeddings = load_embeddings()
        vectordb = SimpleVectorDB.load(DB_PATH, embeddings)
        llm = load_llm()
        logger.info("Resources loaded successfully.")
        return vectordb, llm

    try:
        vectordb, llm = load_resources()
    except Exception as e:
        logger.exception("Failed to load core resources.")
        st.error(f"❌ Failed to load backend: {e}")
        return

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": (
                "🤖 Welcome! I'm your diabetes assistant. Let's personalize your experience.\n"
                "📋 First, do you have any existing conditions I should know about?\n"
                "_For example: Type 1 diabetes, hypertension, obesity, etc._"
            )}
        ]
        logger.info("Chat session started with welcome message.")

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User sends a message
    if query := st.chat_input("Ask your diabetes-related question..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        logger.info(f"User query: {query}")
        with st.chat_message("user"):
            st.markdown(query)

        try:
            retrieval_start = time.time()
            retrieved_docs = custom_retriever(vectordb, query)
            context = "\n".join(retrieved_docs)
            retrieval_time = time.time() - retrieval_start
            logger.info(f"Context retrieved in {retrieval_time:.2f} seconds")

            language_hint = "ES" if any(word in query.lower() for word in ["el", "la", "los", "qué", "cómo"]) else "EN"
            prompt = build_prompt(query, context, patient_conditions, language_hint)
            logger.info(f"Prompt built with {len(prompt)} characters")

            llm_start = time.time()
            response_text = llm(prompt)
            llm_time = time.time() - llm_start
            logger.info(f"LLM response generated in {llm_time:.2f} seconds")

            answer = post_process_response(response_text)
            logger.info(f"Response length: {len(answer)} characters")

        except Exception as e:
            logger.exception(f"Error during response generation: {str(e)}")
            answer = f"❌ Error: {e}"

        # Display and log assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

        logger.info("Response sent to UI.")

if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as e:
        logger.exception(f"Fatal error: {str(e)}")
        print(f"An error occurred. Check the log file at {log_file} for details.")