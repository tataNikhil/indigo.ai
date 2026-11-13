"""Get recommendations from the trained model (run train.py first).

Usage:
    python recommend.py --user 1
    python recommend.py --user 1 --method user
    python recommend.py --new-user
    python recommend.py --similar "Toy Story"
"""

import argparse
import sys

import pandas as pd

from recommender.engine import MODEL_PATH, RecommendationEngine


def show(title, rows):
    print(f"\n{title}")
    if not rows:
        print("No recommendations found.")
        return
    print(pd.DataFrame(rows)[["movieId", "title", "score", "method"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Movie recommendations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user", type=int, help="user id from ratings.csv")
    group.add_argument("--new-user", action="store_true", help="recommendations for a new user")
    group.add_argument("--similar", help="movie title (or part of it)")
    parser.add_argument("--method", choices=["item", "user"], default="item")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run: python train.py")
    engine = RecommendationEngine.load()

    if args.new_user:
        show("Popular movies for a new user", engine.popular_movies(args.k))
    elif args.similar:
        movie_id = engine.find_movie(args.similar)
        if movie_id is None:
            sys.exit(f"No movie matching '{args.similar}'")
        title = engine.movies.loc[movie_id, "title"]
        show(f"Movies similar to {title}", engine.similar_movies(movie_id, args.k))
    else:
        show(f"Top {args.k} for user {args.user} ({args.method}-based)",
             engine.recommend(args.user, args.k, args.method))


if __name__ == "__main__":
    main()
