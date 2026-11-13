import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

MIN_RATING, MAX_RATING = 0.5, 5.0


def _top_k(scores, k):
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


class PopularRecommender:
    """Fallback for new users: well-rated movies with enough ratings.

    Uses a weighted rating so a movie with two 5-star ratings doesn't beat one
    with hundreds of 4.5s.
    """

    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings

    def fit(self, im):
        self.im = im
        counts = np.diff(im.matrix.tocsc().indptr)
        sums = np.asarray(im.matrix.sum(axis=0)).ravel()
        self.item_means = np.where(counts > 0, sums / np.maximum(counts, 1), im.global_mean)
        m, c = self.min_ratings, im.global_mean
        self.scores = (counts / (counts + m)) * self.item_means + (m / (counts + m)) * c
        self.scores[counts < self.min_ratings] = -np.inf
        return self

    def predict(self, user_id, movie_id):
        i = self.im.movie_index.get(movie_id)
        return float(self.item_means[i]) if i is not None else self.im.global_mean

    def recommend(self, user_id=None, k=10):
        scores = self.scores.copy()
        u = self.im.user_index.get(user_id)
        if u is not None:
            scores[self.im.matrix[u].indices] = -np.inf
        top = [i for i in _top_k(scores, k) if np.isfinite(scores[i])]
        return [(int(self.im.movie_ids[i]), float(scores[i])) for i in top]


class ItemBasedCF:
    """Recommends movies similar to the ones a user already rated."""

    def __init__(self, n_neighbors=50, block_size=1000):
        self.n_neighbors = n_neighbors
        self.block_size = block_size

    def fit(self, im):
        self.im = im
        # movie vectors = columns of the user-centered rating matrix (adjusted cosine)
        item_vectors = im.centered().T.tocsr()
        n_items = item_vectors.shape[0]

        # keep only each movie's top neighbours so the similarity matrix stays sparse
        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.block_size):
            block = cosine_similarity(item_vectors[start:start + self.block_size], item_vectors)
            for offset, sims in enumerate(block):
                i = start + offset
                sims[i] = 0
                top = _top_k(sims, self.n_neighbors)
                top = top[sims[top] > 0]
                rows.extend([i] * len(top))
                cols.extend(top)
                vals.extend(sims[top])

        self.similarity = csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)), shape=(n_items, n_items)
        )
        return self

    def predict(self, user_id, movie_id):
        u = self.im.user_index.get(user_id)
        i = self.im.movie_index.get(movie_id)
        if u is None:
            return self.im.global_mean
        mean = self.im.user_means[u]
        if i is None:
            return float(mean)

        user_row = self.im.matrix[u]
        rated = dict(zip(user_row.indices, user_row.data))
        neighbours = self.similarity[i]
        num = den = 0.0
        for j, s in zip(neighbours.indices, neighbours.data):
            if j in rated:
                num += s * (rated[j] - mean)
                den += abs(s)
        pred = mean + num / den if den > 0 else mean
        return float(np.clip(pred, MIN_RATING, MAX_RATING))

    def recommend(self, user_id, k=10):
        u = self.im.user_index.get(user_id)
        if u is None:
            return []
        user_row = self.im.matrix[u]
        scores = np.asarray(self.similarity @ user_row.T.toarray()).ravel()
        scores[user_row.indices] = -np.inf
        top = [i for i in _top_k(scores, k) if scores[i] > 0]
        return [(int(self.im.movie_ids[i]), float(scores[i])) for i in top]

    def similar_movies(self, movie_id, k=10):
        i = self.im.movie_index.get(movie_id)
        if i is None:
            return []
        row = self.similarity[i]
        order = np.argsort(-row.data)[:k]
        return [(int(self.im.movie_ids[row.indices[j]]), float(row.data[j])) for j in order]


class UserBasedCF:
    """Recommends what the most similar users liked."""

    def __init__(self, n_neighbors=30):
        self.n_neighbors = n_neighbors

    def fit(self, im):
        self.im = im
        self.similarity = cosine_similarity(im.centered())
        np.fill_diagonal(self.similarity, 0)
        self.by_movie = im.matrix.tocsc()
        return self

    def predict(self, user_id, movie_id):
        u = self.im.user_index.get(user_id)
        i = self.im.movie_index.get(movie_id)
        if u is None:
            return self.im.global_mean
        mean = self.im.user_means[u]
        if i is None:
            return float(mean)

        col = self.by_movie[:, i]
        raters, ratings = col.indices, col.data
        sims = self.similarity[u, raters]
        keep = _top_k(sims, self.n_neighbors)
        keep = keep[sims[keep] > 0]
        if len(keep) == 0:
            return float(mean)
        deviations = ratings[keep] - self.im.user_means[raters[keep]]
        pred = mean + np.dot(sims[keep], deviations) / np.abs(sims[keep]).sum()
        return float(np.clip(pred, MIN_RATING, MAX_RATING))

    def recommend(self, user_id, k=10):
        u = self.im.user_index.get(user_id)
        if u is None:
            return []
        sims = self.similarity[u]
        neighbours = _top_k(sims, self.n_neighbors)
        neighbours = neighbours[sims[neighbours] > 0]
        if len(neighbours) == 0:
            return []
        scores = np.asarray(sims[neighbours] @ self.im.matrix[neighbours].toarray()).ravel()
        scores[self.im.matrix[u].indices] = -np.inf
        top = [i for i in _top_k(scores, k) if scores[i] > 0]
        return [(int(self.im.movie_ids[i]), float(scores[i])) for i in top]
