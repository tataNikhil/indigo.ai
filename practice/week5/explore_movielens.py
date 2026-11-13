"""Week 5 pre-class task: explore movies.csv and ratings.csv with Pandas.

Usage (from the Indigo.ai folder):
    python practice/week5/explore_movielens.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "movie-recommendation-system" / "data"

movies = pd.read_csv(DATA / "movies.csv")
ratings = pd.read_csv(DATA / "ratings.csv")

print("movies.csv")
print(movies.head(), "\n")
print("ratings.csv")
print(ratings.head(), "\n")

n_users = ratings["userId"].nunique()
n_movies = ratings["movieId"].nunique()
print(f"{len(ratings):,} ratings, {n_users} users, {n_movies:,} rated movies "
      f"(out of {len(movies):,} in movies.csv)")
print(f"Matrix would be {n_users} x {n_movies:,}, only "
      f"{len(ratings) / (n_users * n_movies):.2%} filled -> use a sparse matrix\n")

print("Missing values:", int(movies.isna().sum().sum() + ratings.isna().sum().sum()))
print("Duplicate user/movie pairs:", int(ratings.duplicated(["userId", "movieId"]).sum()), "\n")

print("Rating distribution:")
print(ratings["rating"].value_counts().sort_index().to_string(), "\n")

per_user = ratings.groupby("userId").size()
per_movie = ratings.groupby("movieId").size()
print(f"Ratings per user: min {per_user.min()}, median {per_user.median():.0f}, max {per_user.max()}")
print(f"Ratings per movie: min {per_movie.min()}, median {per_movie.median():.0f}, max {per_movie.max()}")
print(f"Movies with only 1 rating: {(per_movie == 1).sum():,}\n")

genres = movies["genres"].str.split("|").explode().value_counts()
print("Top genres:")
print(genres.head(10).to_string(), "\n")

stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).join(movies.set_index("movieId"))
print("Most rated movies:")
print(stats.sort_values("count", ascending=False).head(10)[["title", "count", "mean"]].to_string())
