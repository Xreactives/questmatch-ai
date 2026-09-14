import os
import numpy as np
import streamlit as st
from src.search_engine import QuestMatcher

st.set_page_config(
    page_title="QuestMatch AI", page_icon="🎮", layout="wide"
)

st.title("🎮 QuestMatch AI")
st.subheader("Semantic Game Recommendation Engine")


@st.cache_resource
def init_engine():
  matcher = QuestMatcher()

  # Load parquet instead of the heavy CSVs
  matcher.initialize_data("data/games_clean.parquet")

  embedding_path = "data/embeddings.npy"
  if os.path.exists(embedding_path):
    matcher.load_saved_embeddings(embedding_path)
  else:
    matcher.build_embeddings(save_path=embedding_path)

  return matcher


engine = init_engine()

if "submitted_query" not in st.session_state:
  st.session_state.submitted_query = ""

# Determine realistic upper bound using 95th percentile (defaults to $60 if array is small)
raw_95th = np.percentile(engine.df["price"], 95)
cap_price = int(raw_95th) if raw_95th > 0 else 100

# --- Sidebar Controls ---
st.sidebar.header("Filter Options")
top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

st.sidebar.subheader("Price Filter")
any_price = st.sidebar.checkbox("No Price Limit", value=True)

if any_price:
  max_price = float("inf")
else:
  max_price = st.sidebar.slider(
      "Max Price ($)", min_value=0, max_value=100, value=100, step=5
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
    filtered_results = [r for r in results if r.get("price", 0) <= max_price][
        :top_k
    ]

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
          st.write(f"**Genres:** {game['genres']}")
          st.write(f"**Developer:** {game['developers']}")

        with col2:
          st.metric(
              label="Match Score", value=f"{game['similarity_score']}%"
          )
          st.write(
              f"**Price:** ${game['price']}"
              if game["price"] > 0
              else "**Free to Play**"
          )