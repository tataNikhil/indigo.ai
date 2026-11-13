import numpy as np
import pandas as pd
import pytest

from recommender.data import InteractionMatrix, clean_data, train_test_split_by_user
from recommender.evaluation import coverage, mae, precision_recall_at_k, rmse
from recommender.models import ItemBasedCF, PopularRecommender, UserBasedCF


@pytest.fixture
def ratings():
    # users 1 and 2 have the same taste; user 3 likes the opposite movies
    rows = [
        (1, 10, 5.0), (1, 20, 4.5), (1, 30, 1.0),
        (2, 10, 5.0), (2, 20, 4.0), (2, 30, 1.5), (2, 40, 5.0),
        (3, 10, 1.0), (3, 30, 5.0), (3, 50, 4.5),
        (4, 10, 4.5), (4, 20, 5.0), (4, 40, 4.5), (4, 50, 1.0),
    ]
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    df["timestamp"] = range(len(df))
    return df


@pytest.fixture
def im(ratings):
    return InteractionMatrix(ratings)


def test_clean_data_drops_bad_rows():
    movies = pd.DataFrame({"movieId": [1, 2], "title": ["A (1999)", "B"], "genres": ["Drama", None]})
    ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2],
        "movieId": [1, 1, 2, 1, 3],
        "rating": [3.0, 4.0, 9.0, 5.0, 4.0],
        "timestamp": [1, 2, 3, 4, 5],
    })
    movies, ratings = clean_data(movies, ratings)

    assert len(ratings) == 2  # duplicate, out-of-range rating and unknown movie removed
    assert ratings.query("userId == 1")["rating"].item() == 4.0  # latest rating kept
    assert movies["year"].tolist()[0] == 1999


def test_split_keeps_every_user_in_train_without_overlap():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "userId": np.repeat([1, 2, 3], 10),
        "movieId": np.tile(range(10), 3),
        "rating": rng.integers(1, 6, 30).astype(float),
    })
    train, test = train_test_split_by_user(df, test_size=0.2)

    assert len(test) == 6
    assert set(train["userId"]) == {1, 2, 3}
    keys = lambda d: set(zip(d["userId"], d["movieId"]))
    assert not keys(train) & keys(test)


def test_interaction_matrix(im):
    assert im.shape == (4, 5)
    assert im.matrix[im.user_index[1], im.movie_index[20]] == 4.5
    assert im.user_means[im.user_index[1]] == pytest.approx(3.5)


def test_item_cf_does_not_recommend_rated_movies(im):
    model = ItemBasedCF(n_neighbors=3).fit(im)
    recs = [m for m, _ in model.recommend(1, k=5)]
    assert recs and not set(recs) & {10, 20, 30}


def test_item_cf_similar_movies(im):
    model = ItemBasedCF(n_neighbors=3).fit(im)
    similar = [m for m, _ in model.similar_movies(10)]
    assert similar[0] == 20  # 10 and 20 are rated alike by everyone
    assert 30 not in similar  # opposite ratings, negative similarity


def test_user_cf_recommends_what_similar_user_liked(im):
    model = UserBasedCF(n_neighbors=2).fit(im)
    assert model.recommend(1, k=1)[0][0] == 40


def test_predictions_stay_in_rating_range(im):
    for model in (ItemBasedCF(n_neighbors=3), UserBasedCF(n_neighbors=2)):
        model.fit(im)
        for movie_id in (10, 20, 30, 40, 50):
            assert 0.5 <= model.predict(1, movie_id) <= 5.0


def test_unknown_user_and_movie_fall_back_to_means(im):
    model = ItemBasedCF().fit(im)
    assert model.recommend(999) == []
    assert model.predict(999, 10) == pytest.approx(im.global_mean)
    assert model.predict(1, 999) == pytest.approx(3.5)


def test_popular_skips_movies_with_few_ratings(im):
    model = PopularRecommender(min_ratings=3).fit(im)
    recs = [m for m, _ in model.recommend(None, k=5)]
    assert set(recs) == {10, 20, 30}
    assert recs[0] == 20


def test_rmse_and_mae():
    assert rmse([4, 2], [3, 4]) == pytest.approx(np.sqrt(2.5))
    assert mae([4, 2], [3, 4]) == pytest.approx(1.5)


def test_precision_recall_and_coverage():
    p, r = precision_recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=4)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(2 / 3)
    assert precision_recall_at_k([1, 2], set(), k=2) == (None, None)
    assert coverage([[1, 2], [2, 3]], catalog_size=6) == pytest.approx(0.5)
