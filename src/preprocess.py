import pandas as pd


def load_parquet_dataset(parquet_path: str) -> pd.DataFrame:
  """Loads the optimized, pre-processed Parquet dataset directly into memory."""
  df = pd.read_parquet(parquet_path)

  # Ensure string types for critical search columns
  df['name'] = df['name'].fillna('Unknown Game').astype(str)
  df['genres'] = df['genres'].fillna('').astype(str)
  df['developers'] = df['developers'].fillna('').astype(str)
  df['corpus'] = df['corpus'].fillna('').astype(str)
  df['price'] = df['price'].fillna(0)

  return df