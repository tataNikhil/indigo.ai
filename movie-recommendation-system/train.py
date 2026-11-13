"""Train the recommenders, evaluate them on held-out ratings and save the model.

Usage:
    python train.py
"""

import time

import pandas as pd

from recommender.data import InteractionMatrix, clean_data, load_data, train_test_split_by_user
from recommender.engine import MODEL_PATH, RecommendationEngine
from recommender.evaluation import evaluate
from recommender.models import ItemBasedCF, PopularRecommender, UserBasedCF

K = 10
MIN_MOVIE_RATINGS = 5


def main():
    movies, ratings = clean_data(*load_data())
    print(f"Ratings: {len(ratings):,}  Users: {ratings['userId'].nunique()}  "
          f"Movies: {ratings['movieId'].nunique():,}")

    # Split first, then build everything from the training part only,
    # so no test rating leaks into the similarities or popularity scores.
    train, test = train_test_split_by_user(ratings, test_size=0.2)

    # movies with very few ratings give unreliable similarities
    counts = train["movieId"].value_counts()
    train = train[train["movieId"].isin(counts[counts >= MIN_MOVIE_RATINGS].index)]
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    im = InteractionMatrix(train)
    density = im.matrix.nnz / (im.shape[0] * im.shape[1])
    print(f"User-item matrix: {im.shape[0]} x {im.shape[1]}, {density:.2%} filled\n")

    models = {
        "Popular (baseline)": PopularRecommender(),
        "Item-based CF": ItemBasedCF(),
        "User-based CF": UserBasedCF(),
    }
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(im)
        results[name] = evaluate(model, test, k=K)
        print(f"{name}: trained and evaluated in {time.time() - start:.1f}s")

    table = pd.DataFrame(results).T.round(4)
    print("\n" + table.to_string())

    engine = RecommendationEngine(
        movies,
        item_cf=models["Item-based CF"],
        user_cf=models["User-based CF"],
        popular=models["Popular (baseline)"],
        metrics=table.to_dict(orient="index"),
    )
    engine.save()
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
