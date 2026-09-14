import os
import urllib.request
import numpy as np
import streamlit as st
from src.search_engine import QuestMatcher

# Direct raw download URL from your Hugging Face dataset
HF_EMBEDDINGS_URL = "https://huggingface.co/datasets/Yreactives/questmatch-embeddings/resolve/main/embeddings.npy"
LOCAL_EMBEDDINGS_PATH = "data/embeddings.npy"

st.set_page_config(
    page_title="QuestMatch AI", page_icon="🎮", layout="wide"
)

st.title("🎮 QuestMatch AI")
st.subheader("Semantic Game Recommendation Engine")


@st.cache_resource
def init_engine():
  matcher = QuestMatcher()
  matcher.initialize_data("data/games_clean.parquet")

  # Download pre-computed embeddings if not present locally
  os.makedirs("data", exist_ok=True)
  if not os.path.exists(LOCAL_EMBEDDINGS_PATH):
    with st.spinner("Fetching vector space from Hugging Face..."):
      urllib.request.urlretrieve(HF_EMBEDDINGS_URL, LOCAL_EMBEDDINGS_PATH)

  matcher.load_saved_embeddings(LOCAL_EMBEDDINGS_PATH)
  return matcher


engine = init_engine()

if "submitted_query" not in st.session_state:
  st.session_state.submitted_query = ""

# --- Sidebar Controls ---
st.sidebar.header("Filter Options")
top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

st.sidebar.subheader("Price Filter")
any_price = st.sidebar.checkbox("No Price Limit", value=True)

if any_price:
  max_price = float("inf")
else:
  max_price = st.sidebar.slider(
      "Max Price ($)", min_value=0, max_value=100, value=60, step=5
  )

# --- Search Input ---
query_input = st.text_input(
    "Describe the kind of game you want to play:",
    placeholder="e.g., cozy roguelike deckbuilder with deep strategy",
)

if st.button("Find Games") and query_input:
  st.session_state.submitted_query = query_input

# --- Render Results ---
if st.session_state.submitted_query:
  with st.spinner("Searching vector space..."):
    results = engine.search(
        st.session_state.submitted_query, top_k=top_k * 5
    )
    filtered_results = [
        r for r in results if float(r.get("price", 0)) <= max_price
    ][:top_k]

    st.markdown(
        f'Showing top results for: *"{st.session_state.submitted_query}"*'
    )

    if not filtered_results:
      st.warning("No games match your current price filter.")
    else:
      for game in filtered_results:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])

        with col1:
          st.markdown(f"### {game['name']}")
          st.write(f"**Genres:** {game.get('genres', 'N/A')}")
          st.write(f"**Developer:** {game.get('developers', 'N/A')}")

        with col2:
          score = int(round(float(game.get("similarity_score", 0))))
          st.metric(label="Match Score", value=f"{score}%")
          price_val = float(game.get("price", 0))
          if price_val > 0:
            st.write(f"**Price:** ${int(round(price_val))}")
          else:
            st.write("**Free to Play**")