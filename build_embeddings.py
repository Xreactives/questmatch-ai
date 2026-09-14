import os
import numpy as np
import pandas as pd
from src.search_engine import QuestMatcher

matcher = QuestMatcher()

# 1. Initialize using your new lightweight Parquet file
parquet_path = "data/games_clean.parquet"
matcher.initialize_data(parquet_path)

# 2. Path to pre-computed embeddings
embedding_path = "data/embeddings.npy"

if not os.path.exists(embedding_path):
  matcher.build_embeddings(save_path=embedding_path)
else:
  print("Loading pre-computed embeddings from disk...")
  matcher.load_saved_embeddings(embedding_path)

# 3. Quick sanity check search
query = "cozy roguelike deckbuilder with deep strategy"
results = matcher.search(query, top_k=3)

print(f"\nResults for: '{query}'")
for game in results:
  print(f"[{game['similarity_score']}% Match] {game['name']} - ${game['price']}")