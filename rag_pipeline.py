from retrieval.search import retrieve_passages
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils.translator import detect_language, translate_to_english, translate_to_spanish

# Load LLM (small BloomZ model for demo; replace with GPT-4 for production)
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU only

def generate_response(query, context):
    prompt = (
        "You are a helpful diabetes management assistant.\n"
        "Use the following medical information to answer the patient's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Patient: {query}\n"
        "Assistant:"
    )
    response = generator(prompt, max_new_tokens=150, do_sample=False)
    return response[0]["generated_text"].split("Assistant:")[-1].strip()

def rag_pipeline(query):
    original_lang = detect_language(query)
    print(f"Detected language: {original_lang}")

    if original_lang != "en":
        query = translate_to_english(query)

    # Step 1: Retrieve
    top_results = retrieve_passages(query, top_k=3)
    context = "\n".join([r[0] for r in top_results])

    # Step 2: Generate
    answer = generate_response(query, context)

    if original_lang != "en":
        answer = translate_to_spanish(answer)

    return answer

# Entry point
if __name__ == "__main__":
    user_question = input("Enter your diabetes question: ")
    response = rag_pipeline(user_question)
    print("\n🩺 Assistant Response:\n", response)
