"""Serving side: loads a trained model and answers recommendation requests."""

import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "recommender.pkl"


class RecommendationEngine:
    def __init__(self, movies, item_cf, user_cf, popular, metrics=None):
        self.movies = movies.set_index("movieId")
        self.models = {"item": item_cf, "user": user_cf}
        self.popular = popular
        self.metrics = metrics or {}

    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=MODEL_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _with_titles(self, recs, method):
        rows = []
        for movie_id, score in recs:
            movie = self.movies.loc[movie_id]
            rows.append({
                "movieId": movie_id,
                "title": movie["title"],
                "genres": movie["genres"],
                "score": round(score, 3),
                "method": method,
            })
        return rows

    def recommend(self, user_id, k=10, method="item"):
        """Personalized picks; new users (or users with no matches) get popular movies."""
        recs = self.models[method].recommend(user_id, k)
        if len(recs) < k:
            seen = {m for m, _ in recs}
            extra = [(m, s) for m, s in self.popular.recommend(user_id, k * 2) if m not in seen]
            return (self._with_titles(recs, method)
                    + self._with_titles(extra[: k - len(recs)], "popular"))
        return self._with_titles(recs, method)

    def popular_movies(self, k=10):
        return self._with_titles(self.popular.recommend(None, k), "popular")

    def find_movie(self, query):
        matches = self.movies[self.movies["title"].str.contains(query, case=False, regex=False)]
        return int(matches.index[0]) if len(matches) else None

    def similar_movies(self, movie_id, k=10):
        """Movies most similar to the given one; popular movies if it has too few ratings."""
        recs = self.models["item"].similar_movies(movie_id, k)
        if not recs:
            return self.popular_movies(k)
        return self._with_titles(recs, "similar")
