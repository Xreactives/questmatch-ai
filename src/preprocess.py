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

def load_and_merge_datasets(
    games_csv_path: str, reviews_csv_path: str
) -> pd.DataFrame:
  """Loads game metadata and review data, cleans missing fields, and merges them into a single processed DataFrame."""
  # 1. Load CSVs
  games_df = pd.read_csv(games_csv_path)
  reviews_df = pd.read_csv(reviews_csv_path)

  # 2. Group reviews by app_id (or name as fallback)
  if "app_id" in reviews_df.columns and "reviews" in reviews_df.columns:
    reviews_grouped = (
        reviews_df.groupby("app_id")["reviews"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
    )
  else:
    reviews_grouped = (
        reviews_df.groupby("name")["reviews"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
    )

  # 3. Merge games metadata with combined reviews
  merge_key = "app_id" if "app_id" in reviews_grouped.columns else "name"
  df = pd.merge(games_df, reviews_grouped, on=merge_key, how="left")

  # 4. Fill missing values to prevent NaN floats
  df["name"] = df["name"].fillna("Unknown Game").astype(str)
  df["genres"] = df["genres"].fillna("").astype(str)
  df["categories"] = df["categories"].fillna("").astype(str)
  df["developers"] = df["developers"].fillna("").astype(str)
  df["reviews"] = df["reviews"].fillna("").astype(str)
  df["price"] = df["price"].fillna(0)

  # 5. Build corpus and force complete string conversion
  df["corpus"] = (
      "Title: "
      + df["name"]
      + " | Genres: "
      + df["genres"]
      + " | Categories: "
      + df["categories"]
      + " | Developer: "
      + df["developers"]
      + " | Player Reviews: "
      + df["reviews"].str[:500]
  )

  # Explicitly cast entire corpus column to string and drop any lingering nulls
  df["corpus"] = df["corpus"].fillna("").astype(str)

  return df