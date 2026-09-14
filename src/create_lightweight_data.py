import os
import pandas as pd
from src.preprocess import load_and_merge_datasets


def prepare_production_data():
  print("Processing 1GB dataset into optimized lightweight file...")

  # 1. Merge and clean raw CSVs
  df = load_and_merge_datasets(
      "../data/steam_games.csv", "../data/steam_games_reviews.csv"
  )

  # 2. Keep ONLY essential columns needed by app.py
  essential_cols = [
      "app_id",
      "name",
      "genres",
      "developers",
      "price",
      "corpus",
  ]
  df_light = df[essential_cols].copy()

  # 3. Save as compressed Parquet (drastically smaller and faster to load)
  os.makedirs("../data", exist_ok=True)
  output_path = "../data/games_clean.parquet"
  df_light.to_parquet(output_path, compression="snappy", index=False)

  file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
  print(f"Success! Optimized dataset size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
  prepare_production_data()