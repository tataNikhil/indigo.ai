import numpy as np


def rmse(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def precision_recall_at_k(recommended, relevant, k):
    """recommended: list of movie ids in rank order; relevant: set of movie ids."""
    if not relevant:
        return None, None
    hits = len(set(recommended[:k]) & relevant)
    return hits / k, hits / len(relevant)


def coverage(recommendation_lists, catalog_size):
    """Share of the catalog that shows up in at least one user's recommendations."""
    recommended = {m for recs in recommendation_lists for m in recs}
    return len(recommended) / catalog_size if catalog_size else 0.0


def evaluate(model, test, k=10, relevance_threshold=4.0):
    """Rating accuracy (RMSE/MAE) and ranking quality (Precision@K, Recall@K, coverage).

    `test` must only contain ratings the model never saw during fit.
    """
    predictions = [model.predict(u, m) for u, m in zip(test["userId"], test["movieId"])]

    liked = test[test["rating"] >= relevance_threshold].groupby("userId")["movieId"].apply(set)
    precisions, recalls, all_recs = [], [], []
    for user_id, relevant in liked.items():
        recs = [movie_id for movie_id, _ in model.recommend(user_id, k)]
        all_recs.append(recs)
        p, r = precision_recall_at_k(recs, relevant, k)
        precisions.append(p)
        recalls.append(r)

    return {
        "rmse": rmse(test["rating"], predictions),
        "mae": mae(test["rating"], predictions),
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        "coverage": coverage(all_recs, model.im.shape[1]),
    }
