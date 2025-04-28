# Mental Health Utterance Prediction

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline using a fine-tuned FLAN-T5 model to predict supportive mental health utterances based on user input. The core logic and experiments are contained in `main.ipynb`.

---

## Project Overview

With rising awareness around mental health, automated supportive dialogue systems can provide immediate empathetic responses. In this work, we build a pipeline that:

1. **Indexes** a knowledge base of mental health utterances.
2. **Retrieves** the most relevant contexts given a user input.
3. **Generates** a supportive utterance using a fine-tuned FLAN-T5 model in a RAG framework.

All experiments and code reside in the `main.ipynb` notebook.

---

## Repository Structure

```
├── README.md               # Project overview and instructions
├── main.ipynb              # Jupyter notebook with full pipeline
└── (future files)          # Data, scripts, checkpoints, etc.
```

Detailed breakdown of notebook cells:

- **0. Setup**: install dependencies, import libraries
- **1. Load Data**: read utterance corpus and any labels
- **2. Preprocessing**: text cleaning, tokenization
- **3. Embedding & Index**: generate embeddings (e.g., using SentenceTransformers) and build FAISS index
- **4. RAG Configuration**: load base FLAN-T5, configure retriever and generator
- **5. Fine-tuning**: train model on question–utterance pairs
- **6. Evaluation**: compute BLEU, ROUGE, human eval snippets
- **7. Inference**: sample predictions for sample prompts

---

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/devyansh22156/Mental-Health-Utterance-Prediction.git
   cd Mental-Health-Utterance-Prediction
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the notebook**:
   ```bash
   jupyter notebook main.ipynb
   ```

---

## Data

*Currently, no external data is included.*

- **Utterance Corpus**: a collection of supportive mental health responses used as the knowledge base. (To be added as `data/utterances.csv`)

---

## Methodology

### 1. Data Preprocessing
- Clean and normalize text (lowercasing, removing punctuation).
- Optional stopword removal.

### 2. Embedding & Retrieval
- Compute dense embeddings (e.g., with `sentence-transformers`).
- Build a FAISS index for nearest-neighbor search.
- Given a user query, retrieve top-K relevant utterances.

### 3. RAG Model Fine-tuning
- Use Hugging Face’s `RagTokenForGeneration` with a FLAN-T5 backbone.
- Plug in our custom retriever over the FAISS index.
- Fine-tune on pairs of (prompt, target utterance).

### 4. Inference & Evaluation
- Generate responses for held-out prompts.
- Evaluate with automatic metrics (BLEU, ROUGE).
- Qualitative analysis of model suggestions.

---

## Results

| Metric | Score |
|--------|-------|
| BLEU   | 0.0380  |
| ROUGE-L| 0.8426  |

> *Results placeholder; populate after running fine-tuning.*

---

## Usage

Once fine-tuned, you can load the model and retriever for inference:

```python
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Load tokenizer and model
tokenizer = RagTokenizer.from_pretrained("./checkpoints/flan-t5-rag")
retriever = RagRetriever.from_pretrained(
    pretrained_model_name_or_path="./checkpoints/flan-t5-rag",
    index_name="custom",
    passages_path="data/utterances.csv"
)
model = RagTokenForGeneration.from_pretrained("./checkpoints/flan-t5-rag", retriever=retriever)

# Generate
inputs = tokenizer("I feel anxious and alone", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

---

## Requirements

- Python 3.8+
- Jupyter Notebook
- `torch`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `pandas`, `numpy`, `scikit-learn`

*(Consider freezing versions in a `requirements.txt` file.)*

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for enhancements, bug fixes, or new features.

---

## License

This project is released under the [MIT License](LICENSE).

