from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir=DATA_DIR):
    movies = pd.read_csv(Path(data_dir) / "movies.csv")
    ratings = pd.read_csv(Path(data_dir) / "ratings.csv")
    return movies, ratings


def clean_data(movies, ratings):
    movies = movies.dropna(subset=["movieId", "title"]).drop_duplicates("movieId").copy()
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings = ratings[ratings["rating"].between(0.5, 5.0)]
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
    # if a user rated the same movie twice, keep the latest rating
    ratings = ratings.sort_values("timestamp").drop_duplicates(["userId", "movieId"], keep="last")
    return movies, ratings.reset_index(drop=True)


def train_test_split_by_user(ratings, test_size=0.2, min_ratings=5, seed=42):
    """Hold out a share of every user's ratings, so each user appears in both sets."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, group in ratings.groupby("userId"):
        if len(group) < min_ratings:
            continue
        n_test = int(round(len(group) * test_size))
        test_idx.extend(rng.choice(group.index, size=n_test, replace=False))

    test = ratings.loc[test_idx]
    train = ratings.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


class InteractionMatrix:
    """Sparse user x movie rating matrix with id <-> index lookups."""

    def __init__(self, ratings):
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(ratings["movieId"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}

        rows = ratings["userId"].map(self.user_index).to_numpy()
        cols = ratings["movieId"].map(self.movie_index).to_numpy()
        self.matrix = csr_matrix(
            (ratings["rating"].to_numpy(dtype=np.float32), (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        counts = np.diff(self.matrix.indptr)
        self.user_means = np.asarray(self.matrix.sum(axis=1)).ravel() / np.maximum(counts, 1)
        self.global_mean = float(ratings["rating"].mean())

    @property
    def shape(self):
        return self.matrix.shape

    def centered(self):
        """Ratings minus each user's mean (only on rated cells)."""
        centered = self.matrix.copy().astype(np.float32)
        counts = np.diff(centered.indptr)
        centered.data -= np.repeat(self.user_means, counts).astype(np.float32)
        return centered
