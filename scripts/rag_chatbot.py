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
import re

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
    """
    Retrieve up to k docs, but fall back gracefully if scores are low.
    """
    logger.info(f"Processing query: '{query}'")

    # Pull a few more candidates for safety
    results = vectordb.similarity_search(query, k=10)

    # Log the top‑5 scores for debugging
    if results:
        logger.info(
            "Top retrieval scores: " +
            ", ".join(f"{i+1}:{r.get('score', 0):.4f}" for i, r in enumerate(results[:5]))
        )

    # Keep anything above 0.40 … but never return <2 docs.
    filtered = [r for r in results if r.get("score", 0) >= 0.40] or results[:2]

    # Extract text
    docs = []
    for r in filtered:
        if isinstance(r, dict) and "text" in r:
            docs.append(r["text"])
        elif hasattr(r, "page_content"):
            docs.append(r.page_content)
        elif hasattr(r, "text"):
            docs.append(r.text)
        else:
            docs.append(str(r))

    return docs

def post_process_response(raw: str) -> str:
    """
    • Strips prompt leakage / instruction echoes
    • Removes unfinished trailing sentence
    """
    #  1 Remove any lines that start with '•', '-', or numerals followed by guidelines
    cleaned = re.sub(r"^[\-\d\•].*guidelines?:.*$", "", raw, flags=re.IGNORECASE | re.MULTILINE)

    #  2 Kill everything after an obviously truncated word at the very end
    sentences = re.split(r"(?<=[\.\!\?])\s+", cleaned.strip())
    if sentences and not sentences[-1][-1] in ".!?":
        sentences = sentences[:-1]

    return " ".join(sentences).strip()


# --- Conversational Bot ---
def run_chatbot():
    logger.info("Starting RAG chatbot application")
    log_system_info()
    
    logger.info(f"Loading vector DB from {DB_PATH}...")
    start_time = time.time()
    embeddings = load_embeddings()
    vectordb = SimpleVectorDB.load(DB_PATH, embeddings)
    end_time = time.time()
    
    # Get the size of the vector DB
    try:
        if hasattr(vectordb, "__len__"):
            db_size = len(vectordb)
        elif hasattr(vectordb, "size"):
            db_size = vectordb.size
        else:
            db_size = "unknown"
    except:
        db_size = "unknown"
    
    logger.info(f"Vector DB loaded with {db_size} texts in {end_time - start_time:.2f} seconds")
    
    llm = load_llm()
    
    # Keep statistics for session
    session_stats = {
        "queries": 0,
        "total_retrieval_time": 0,
        "total_llm_time": 0
    }
    
    # Initialize patient context with user information
    patient_context = {}
    print("\n🤖 Welcome! I'm your diabetes assistant. Let's personalize your experience.")
    print("📋 First, do you have any existing conditions I should know about?")
    print("For example: Type 1 diabetes, hypertension, obesity, etc.")

    condition_input = input("\nYou: ")
    patient_context["conditions"] = condition_input.strip()

    print("\n✅ Got it. You can now ask me any diabetes-related question.")
    print("Type 'exit' anytime to quit.\n")
    
    logger.info("RAG chatbot system initialized and ready for queries")
    
    while True:
        try:
            user_query = input("You: ").strip()
            if user_query.lower() in ["exit", "quit"]:
                logger.info("User ended session")
                print("👋 Goodbye!")
                
                # Log session statistics
                if session_stats["queries"] > 0:
                    avg_retrieval = session_stats["total_retrieval_time"] / session_stats["queries"]
                    avg_llm = session_stats["total_llm_time"] / session_stats["queries"]
                    logger.info(f"Session summary: {session_stats['queries']} queries processed")
                    logger.info(f"Average retrieval time: {avg_retrieval:.4f} seconds")
                    logger.info(f"Average LLM processing time: {avg_llm:.4f} seconds")
                break

            logger.info(f"New user query: '{user_query}'")
            session_stats["queries"] += 1

            # Time the retrieval process
            retrieval_start = time.time()
            retrieved_docs = custom_retriever(vectordb, user_query)
            retrieval_end = time.time()
            retrieval_time = retrieval_end - retrieval_start
            session_stats["total_retrieval_time"] += retrieval_time
            
           # --- Build prompt ----------------------------------------------------
            language_hint = "ES" if re.search(r"\b(el|la|los|las|qué|cómo|para|dónde|por)\b",
                                            user_query.lower()) else "EN"

            prompt = f"""You are a medically‑accurate diabetes assistant.
            Respond in the same language as the question ({'Spanish' if language_hint=='ES' else 'English'}).

            PATIENT CONDITIONS: {patient_context['conditions']}
            RETRIEVED INFO:
            {context}

            QUESTION: {user_query}

            ANSWER:"""


            # Log prompt length
            logger.info(f"Prompt length: {len(prompt)} characters")

            # Time the LLM processing
            llm_start = time.time()
            result = llm(prompt)
            answer = post_process_response(result)

            print(f"\n🤖 Assistant: {answer}\n")
            llm_end = time.time()
            llm_time = llm_end - llm_start
            session_stats["total_llm_time"] += llm_time
            
            logger.info(f"Generated response in {llm_time:.4f} seconds")
            logger.info(f"Response length: {len(result)} characters")
            
            
            
            # Log performance metrics
            logger.info(f"Query performance - Retrieval: {retrieval_time:.4f}s, LLM: {llm_time:.4f}s, Total: {retrieval_time + llm_time:.4f}s")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print("Sorry, I encountered an error. Please try again.")



if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as e:
        logger.exception(f"Fatal error: {str(e)}")
        print(f"An error occurred. Check the log file at {log_file} for details.")