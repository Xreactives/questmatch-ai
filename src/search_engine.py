import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import load_parquet_dataset


class QuestMatcher:

  def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
    self.model = SentenceTransformer(model_name)
    self.df = None
    self.embeddings = None

  def initialize_data(self, parquet_path: str):
    """Loads dataset directly from the optimized Parquet file."""
    print(f'Loading optimized Parquet dataset from {parquet_path}...')
    self.df = load_parquet_dataset(parquet_path)

  def load_saved_embeddings(self, filepath: str):
    """Loads pre-computed numpy embeddings directly into memory."""
    self.embeddings = np.load(filepath)

  def build_embeddings(self, save_path: str = None):
    print("Generating vector embeddings...")
    corpus_list = [str(text) for text in self.df['corpus'].tolist()]
    self.embeddings = self.model.encode(
      corpus_list, show_progress_bar=True, batch_size=32
    )

    if save_path:
      np.save(save_path, self.embeddings)

  def search(self, query: str, top_k: int = 5):
    if self.embeddings is None:
      raise ValueError(
        "Embeddings not loaded! Call load_saved_embeddings() before searching."
      )
    query_vector = self.model.encode([query])
    similarities = cosine_similarity(query_vector, self.embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
      game_data = self.df.iloc[idx].to_dict()
      game_data['similarity_score'] = int(round(similarities[idx] * 100))
      results.append(game_data)

    return results