# 🎮 QuestMatch AI

**Semantic Game Recommendation Engine**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://questmatch-ai-dbsdfofdodtarthpva2pz3.streamlit.app/)

QuestMatch AI is a natural language recommendation engine for video games. Unlike traditional tag-based filters, QuestMatch AI uses vector embeddings to understand the **semantic meaning** of user prompts—matching players with games based on complex descriptions, mechanics, and vibes.

---

## 🚀 Live Demo

Try the live app on Streamlit Cloud:  
👉 **[questmatch-ai.streamlit.app](https://questmatch-ai-dbsdfofdodtarthpva2pz3.streamlit.app/)**

---

## ✨ Features

- **Semantic Search:** Describe your ideal game in natural language (e.g., *"cozy roguelike deckbuilder with deep strategy"*).
- **Instant Similarity Scoring:** Uses cosine similarity over pre-computed high-dimensional embeddings.
- **Dynamic Price Filtering:** Filter search results by price range or "Free to Play" options.
- **Cloud-Optimized Architecture:** Offloads heavy vector generation to external datasets (Hugging Face Hub) for lightning-fast application boot times under strict 1 GB RAM constraints.

---

## 🏗️ Architecture & Technical Stack

```
           [ User Query ]
                 │
                 ▼
  SentenceTransformer (all-MiniLM-L6-v2)
                 │
                 ▼ (1x384 Vector)
  Cosine Similarity Search vs pre-computed embeddings.npy
                 │
                 ▼
     Streamlit Interactive UI
```

- **Frontend / Framework:** Streamlit
- **Embeddings Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Operations:** NumPy & Scikit-Learn (Cosine Similarity)
- **Data Format:** Apache Parquet (compressed metadata storage)
- **External Asset Storage:** Hugging Face Hub Datasets (Vector array streaming)

---

## 🛠️ Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Yreactives/questmatch-ai.git
cd questmatch-ai
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Vectors & Run App
```bash
# Optional: Pre-compute vector embeddings locally
python build_embeddings.py

# Launch Streamlit app
streamlit run app.py
```

