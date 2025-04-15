Sure! Here's a complete and copy-ready `README.md` tailored to your **CSE 635 Phase 2 project**, reflecting all the structure, components, and usage details:

---

```markdown
# 🩺 Multilingual Health Chatbot for Diabetes Management (CSE 635 Project Phase 2)

This project aims to build an intelligent multilingual health chatbot to assist users with **diabetes-related medical queries**, using **Retrieval-Augmented Generation (RAG)** and **LoRA fine-tuning on BLOOMZ**.

---

## 📚 Problem Statement

Managing diabetes involves numerous patient queries around symptoms, medications, glucose levels, lifestyle, and complications. With the rise of LLMs, this project explores creating a chatbot that:

- Answers diabetes-related questions reliably and factually.
- Works across **English** and **multilingual** settings.
- Can be fine-tuned to better understand **domain-specific language** and **contextual responses** using medical QA datasets.

---

## 📦 Project Structure

```
CSE635-project/
├── data/                            # Raw + filtered datasets
│   └── prepared/                   # Diabetes-specific QA pairs (MedQuAD, ReMeDi, etc.)
├── evaluation/                     # Evaluation scripts + metrics
├── lora-bloomz-diabetes/          # Trained LoRA adapter checkpoints
├── preparation/                    # Dataset preparation scripts
├── scripts/                        # Fine-tuning, filtering, and inference scripts
├── vectordb/                       # Vector store generation and document chunking
├── scraped_text.json              # Raw medical corpus for vector store
├── README.md                       # ✅ You are here
├── requirements.txt
├── environment.yml
└── setup.sh
```

---

## 🧠 Datasets Used

We combined and filtered several publicly available QA and dialogue datasets for **diabetes-specific questions**:

| Dataset        | Type        | Language | Count |
|----------------|-------------|----------|-------|
| MedQuAD        | QA          | English  | 2128  |
| ReMeDi         | Dialogue    | Chinese  | 52    |
| LLM-CGM        | QA          | English  | 30    |
| HealthSearchQA | QA          | English  | 42    |
| Manual QA set  | QA          | English  | 74    |

> All filtered datasets are saved in `data/prepared/`.

---

## 🧱 Solution Architecture

### 🧩 Retrieval-Augmented Generation (RAG)

- Documents from `scraped_text.json` are:
  - Split into semantically coherent chunks using **cosine distance-based segmentation**.
  - Embedded using `neuml/pubmedbert-base-embeddings`.
  - Indexed for retrieval using **FAISS**.

### 🔧 Fine-tuning (LoRA on BLOOMZ)

- Base model: `bigscience/bloomz-560m`
- Fine-tuned using LoRA on combined 1800+ diabetes QAs.
- Frameworks: `PEFT`, `transformers`, `accelerate`
- Saved at: `lora-bloomz-diabetes/`

---

## 🧪 Evaluation

### Metrics (on MedQuAD dev subset, N = 50):

| Metric        | Score   |
|---------------|---------|
| BLEU          | 0.0000  |
| ROUGE-1       | 0.1067  |
| ROUGE-2       | 0.0431  |
| ROUGE-L       | 0.0912  |
| BERTScore-F1  | 0.8285  |

> 📌 Despite low lexical overlap (BLEU/ROUGE), BERTScore shows strong **semantic similarity**, indicating medically relevant answers.

---

## 🌍 Multilingual Support

- **ReMeDi (Chinese)** dialogues were included.
- Planned extension to **Spanish** via translation using `MarianMT` or `googletrans`.
- Tokenizer: `bloomz` supports multilingual prompting.

---

## 🚀 Setup Instructions

### 🔧 Option 1: Using Conda

```bash
conda env create -f environment.yml
conda activate diabetes-chatbot
python -m spacy download en_core_web_sm
```

### ⚙️ Option 2: Manual Setup

```bash
bash setup.sh
```

---

## 🛠️ Run Instructions

### 🔹 Prepare datasets
```bash
python preparation/prepare_all_datasets.py
```

### 🔹 Filter diabetes-specific QAs
```bash
python scripts/filter_diabetes_qa.py
```

### 🔹 Fine-tune model with LoRA
```bash
python scripts/finetune_lora_bloomz.py
```

### 🔹 Evaluate fine-tuned model
```bash
python scripts/evaluate_finetuned_lora.py
```

### 🔹 Build document vector store (optional RAG)
```bash
cd vectordb
bash run_batches.sh
```

---

## 📌 Future Plans

- Scale up to `bloomz-1b1` or `3b` for better performance.
- Add **few-shot examples** or better system prompts.
- Integrate **RAG pipeline end-to-end** for real-time doc retrieval + generation.
- Deploy chatbot via **Gradio/Streamlit**.

---

## ✨ Contributors

- **Shubhang Kohirkar** – Fine-tuning, evaluation, multilingual setup
- **Team Member 2** – Document chunking & vector store (see `vectordb/`)

---

## 📄 License

MIT License © 2025 | Built as part of CSE 635 Project at Northeastern University

```

---

Let me know if you want to personalize this more (like logo, screenshots, or citation).