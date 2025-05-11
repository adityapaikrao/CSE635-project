# 🩺 Multilingual Health Chatbot for Diabetes Management

**CSE 635 Project Phase 2**

![Project Banner](https://img.shields.io/badge/Medical-Chatbot-blue) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow) ![Language](https://img.shields.io/badge/Multilingual-Support-green)

> An intelligent multilingual health chatbot to assist users with **diabetes-related medical queries**, using **Retrieval-Augmented Generation (RAG)** and **LoRA fine-tuning on BLOOMZ**.

## 📚 Problem Statement

Managing diabetes involves numerous patient queries around symptoms, medications, glucose levels, lifestyle, and complications. With the rise of LLMs, this project explores creating a chatbot that:

- Answers diabetes-related questions reliably and factually
- Works across **English** and **multilingual** settings
- Can be fine-tuned to better understand **domain-specific language** and **contextual responses** using medical QA datasets

## 📦 Project Structure

```
CSE635-project/
├── data/                           # Raw + filtered datasets
│   └── prepared/                   # Diabetes-specific QA pairs (MedQuAD, ReMeDi, etc.)
├── evaluation/                     # Evaluation scripts + metrics
├── lora-bloomz-diabetes/           # Trained LoRA adapter checkpoints
├── preparation/                    # Dataset preparation scripts
├── scripts/                        # Fine-tuning, filtering, and inference scripts
├── vectordb/                       # Vector store generation and document chunking
├── scraped_text.json               # Raw medical corpus for vector store
├── README.md                       # 📋 You are here
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment specification
└── setup.sh                        # Automated setup script
```

## 🧠 Datasets Used

We combined and filtered several publicly available QA and dialogue datasets for **diabetes-specific questions**:

| Dataset | Type | Language | Count |
|---------|------|----------|-------|
| MedQuAD | QA | English | 2128 |
| ReMeDi | Dialogue | Chinese | 52 |
| LLM-CGM | QA | English | 30 |
| HealthSearchQA | QA | English | 42 |
| Manual QA set | QA | English | 74 |

> All filtered datasets are saved in `data/prepared/`

## 🧱 Solution Architecture

### 🔍 Retrieval-Augmented Generation (RAG)

- **Document Processing**: Medical texts from `scraped_text.json` are:
  - Split into semantically coherent chunks using **cosine distance-based segmentation**
  - Embedded using `neuml/pubmedbert-base-embeddings`
  - Indexed for retrieval using **FAISS**

### 🔧 Fine-tuning (LoRA on BLOOMZ)

- **Base Model**: `bigscience/bloomz-560m`
- **Technique**: Fine-tuned using LoRA on combined 1800+ diabetes QAs
- **Frameworks**: `PEFT`, `transformers`, `accelerate`
- **Checkpoint Location**: `lora-bloomz-diabetes/`

## 🧪 Evaluation Results

### Metrics (on MedQuAD dev subset, N = 50):

| Metric | Score |
|--------|-------|
| BLEU | 0.0000 |
| ROUGE-1 | 0.1067 |
| ROUGE-2 | 0.0431 |
| ROUGE-L | 0.0912 |
| BERTScore-F1 | 0.8285 |

> 📌 Despite low lexical overlap (BLEU/ROUGE), BERTScore shows strong **semantic similarity**, indicating medically relevant answers.

## 🌍 Multilingual Support

- **Chinese**: Integration through ReMeDi dialogues
- **Spanish**: Planned extension via translation using `MarianMT` or `googletrans`
- **Tokenization**: `bloomz` natively supports multilingual prompting

## 🚀 Setup Instructions

### Option 1: Using Conda

```bash
conda env create -f environment.yml
conda activate diabetes-chatbot
python -m spacy download en_core_web_sm
```

### Option 2: Manual Setup

```bash
bash setup.sh
```

## 🛠️ Usage Instructions

### Prepare Datasets
```bash
python preparation/prepare_all_datasets.py
```

### Filter Diabetes-specific QAs
```bash
python scripts/filter_diabetes_qa.py
```

### Fine-tune Model with LoRA
```bash
python scripts/finetune_lora_bloomz.py
```

### Evaluate Fine-tuned Model
```bash
python scripts/evaluate_finetuned_lora.py
```

### Build Document Vector Store (for RAG)
```bash
cd vectordb
bash run_batches.sh
```
# 🚀 Deployment Guide for Multilingual Diabetes Chatbot

This guide helps you deploy the full chatbot app via **Streamlit Cloud** or **Render.com**.

---

## 🌐 Deployment Options

### ✅ Streamlit Cloud (Recommended for demo/testing)

1. **Push to GitHub**
   Make sure your repo includes:
   - `full_chatbot_app.py`
   - `requirements.txt`
   - `Procfile`
   - `utils/`, `scripts/`, `data/` (but NO large LoRA or model files)

2. **Remove Large Files Before Commit**
   ```bash
   rm -rf lora-bloomz-diabetes/ models/ checkpoints/ vectordb/
   echo "lora-bloomz-diabetes/\nmodels/\ncheckpoints/\nvectordb/" >> .gitignore
   ```

3. **Create GitHub Repo**
   ```bash
   git init
   git remote add origin https://github.com/yourusername/diabetes-chatbot
   git add .
   git commit -m "Initial deployable version"
   git push -u origin main
   ```

4. **Deploy on Streamlit Cloud**
   - Visit [streamlit.io/cloud](https://streamlit.io/cloud)
   - Click **New App** → Select your repo
   - Set `full_chatbot_app.py` as the entry file
   - Click **Deploy** 🎉

---

### ✅ Render.com (For flexible web deployment)

1. **Same GitHub Setup as above**
2. **Add the following files (already included):**
   - `requirements.txt`
   - `Procfile`

3. **Render Setup**
   - Go to [https://render.com](https://render.com)
   - Click "New Web Service"
   - Connect your GitHub
   - Choose:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run full_chatbot_app.py --server.port=$PORT`
     - **Environment**: Python 3.10+

---

## 📦 Deployment File Summary

### `requirements.txt`
```text
streamlit==1.32.2
torch>=2.0
transformers
sentencepiece
langchain>=0.1.0
langchain-community
bert-score
gtts
SpeechRecognition
soundfile
matplotlib
pandas
tqdm
scikit-learn
huggingface_hub
openpyxl
python-dotenv
```

### `Procfile`
```bash
web: streamlit run full_chatbot_app.py --server.port=$PORT
```

---

## 📁 Folder Structure for Deployment (Minimal)
```
CSE635-project/
├── full_chatbot_app.py
├── requirements.txt
├── Procfile
├── utils/
│   ├── patient_profiles.py
│   └── translator.py
├── scripts/
│   └── rag_chatbot.py
├── data/
│   └── alignment_dataset.json
├── logs/              # (optional, gitignore it)
│   └── chat_history.db
└── README.md
```

> ⚠️ Exclude large model weights (`lora-bloomz-diabetes/`, `vectordb/`) from GitHub to meet cloud limits.

---

## ✅ Next Steps
- Test your app at: `https://your-app-name.streamlit.app`
- Share the public link with evaluators/instructors
- Continue training locally and push lighter endpoints for inference

---

Need help with Hugging Face Spaces or Docker deployment? Let us know!

## ✨ Contributors

- **Aditya** – 
- **Atul** – 

## 📄 License

MIT License © 2025 | Built as part of CSE 635 Project at University at Buffalo