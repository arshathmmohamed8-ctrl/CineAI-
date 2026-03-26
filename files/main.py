from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import json
from typing import Optional

app = FastAPI(title="CineAI Movie API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────

POPULAR_GENRES = {"action", "drama", "science fiction", "sci-fi", "thriller", "comedy", "adventure"}

df: Optional[pd.DataFrame] = None
tfidf_matrix = None
tfidf_vectorizer = None


def parse_genres(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [g.get("name", "") if isinstance(g, dict) else str(g) for g in val]
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [g.get("name", "") if isinstance(g, dict) else str(g) for g in parsed]
    except Exception:
        pass
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return []


def parse_languages(val):
    if pd.isna(val):
        return "Unknown"
    if isinstance(val, list):
        langs = [l.get("name", "") if isinstance(l, dict) else str(l) for l in val]
        return ", ".join(langs) if langs else "Unknown"
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            langs = [l.get("name", "") if isinstance(l, dict) else str(l) for l in parsed]
            return ", ".join(langs) if langs else "Unknown"
    except Exception:
        pass
    return str(val) if val else "Unknown"


def load_dataset():
    global df, tfidf_matrix, tfidf_vectorizer

    # Try common TMDB / MovieLens CSV paths
    candidates = [
        "movies_metadata.csv",
        "movies.csv",
        "dataset.csv",
        "data/movies_metadata.csv",
        "data/movies.csv",
    ]

    loaded = False
    for path in candidates:
        if os.path.exists(path):
            try:
                raw = pd.read_csv(path, low_memory=False)
                raw.columns = raw.columns.str.lower().str.strip()

                col_map = {}
                for col in raw.columns:
                    if col in ("title", "original_title"):
                        col_map.setdefault("title", col)
                    if col in ("overview", "description", "plot"):
                        col_map.setdefault("overview", col)
                    if col in ("genres",):
                        col_map.setdefault("genres", col)
                    if col in ("release_date", "year", "release_year"):
                        col_map.setdefault("release_date", col)
                    if col in ("spoken_languages", "language", "original_language"):
                        col_map.setdefault("language", col)

                if "title" not in col_map or "overview" not in col_map:
                    continue

                df = raw.rename(columns={v: k for k, v in col_map.items()})

                # Keep only needed columns
                keep = [c for c in ["title", "overview", "genres", "release_date", "language"] if c in df.columns]
                df = df[keep].copy()

                df["title"] = df["title"].fillna("").astype(str)
                df["overview"] = df["overview"].fillna("").astype(str)

                if "genres" in df.columns:
                    df["genres_list"] = df["genres"].apply(parse_genres)
                else:
                    df["genres_list"] = [[]] * len(df)

                if "language" in df.columns:
                    df["language_str"] = df["language"].apply(parse_languages)
                else:
                    df["language_str"] = "Unknown"

                if "release_date" in df.columns:
                    df["year"] = (
                        pd.to_datetime(df["release_date"], errors="coerce")
                        .dt.year.fillna(0)
                        .astype(int)
                    )
                else:
                    df["year"] = 0

                df = df[df["overview"].str.len() > 20].reset_index(drop=True)
                df = df[df["title"].str.len() > 0].reset_index(drop=True)

                # Build TF-IDF
                tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)
                tfidf_matrix = tfidf_vectorizer.fit_transform(df["overview"])

                loaded = True
                print(f"✅ Loaded {len(df)} movies from '{path}'")
                break
            except Exception as e:
                print(f"⚠️  Failed to load '{path}': {e}")

    if not loaded:
        print("⚠️  No CSV found. Using built-in sample dataset.")
        _load_sample_dataset()


def _load_sample_dataset():
    global df, tfidf_matrix, tfidf_vectorizer

    sample = [
        {
            "title": "Inception",
            "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
            "genres_list": ["Action", "Science Fiction", "Thriller"],
            "year": 2010,
            "language_str": "English",
        },
        {
            "title": "The Dark Knight",
            "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
            "genres_list": ["Action", "Drama", "Crime"],
            "year": 2008,
            "language_str": "English",
        },
        {
            "title": "Interstellar",
            "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival. The film explores themes of time dilation, relativity, and human connection across vast distances.",
            "genres_list": ["Adventure", "Drama", "Science Fiction"],
            "year": 2014,
            "language_str": "English",
        },
        {
            "title": "Parasite",
            "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
            "genres_list": ["Comedy", "Thriller", "Drama"],
            "year": 2019,
            "language_str": "Korean",
        },
        {
            "title": "The Matrix",
            "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers. The film is a groundbreaking sci-fi action thriller.",
            "genres_list": ["Action", "Science Fiction"],
            "year": 1999,
            "language_str": "English",
        },
        {
            "title": "Avengers: Endgame",
            "overview": "After the devastating events of Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe.",
            "genres_list": ["Action", "Adventure", "Science Fiction"],
            "year": 2019,
            "language_str": "English",
        },
        {
            "title": "Pulp Fiction",
            "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
            "genres_list": ["Crime", "Drama"],
            "year": 1994,
            "language_str": "English",
        },
        {
            "title": "The Shawshank Redemption",
            "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency. A story of hope, perseverance, and the indomitable human spirit.",
            "genres_list": ["Drama"],
            "year": 1994,
            "language_str": "English",
        },
        {
            "title": "Dune",
            "overview": "Feature adaptation of Frank Herbert's science fiction novel about the son of a noble family entrusted with the protection of the most valuable asset and most vital element in the galaxy.",
            "genres_list": ["Science Fiction", "Adventure"],
            "year": 2021,
            "language_str": "English",
        },
        {
            "title": "Everything Everywhere All at Once",
            "overview": "An aging Chinese immigrant is swept up in an insane adventure, where she alone can save the world by exploring other universes connecting with the lives she could have led.",
            "genres_list": ["Action", "Adventure", "Science Fiction"],
            "year": 2022,
            "language_str": "English",
        },
        {
            "title": "Oppenheimer",
            "overview": "The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb during World War II. A biographical thriller about power, responsibility, and consequence.",
            "genres_list": ["Drama", "History", "Thriller"],
            "year": 2023,
            "language_str": "English",
        },
        {
            "title": "The Godfather",
            "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son. A masterpiece of American cinema.",
            "genres_list": ["Crime", "Drama"],
            "year": 1972,
            "language_str": "English",
        },
        {
            "title": "Blade Runner 2049",
            "overview": "Young Blade Runner K's discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard, who's been missing for thirty years. A visually stunning neo-noir science fiction film.",
            "genres_list": ["Science Fiction", "Drama"],
            "year": 2017,
            "language_str": "English",
        },
        {
            "title": "Mad Max: Fury Road",
            "overview": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners, a psychotic worshiper, and a drifter named Max.",
            "genres_list": ["Action", "Adventure", "Science Fiction"],
            "year": 2015,
            "language_str": "English",
        },
        {
            "title": "Whiplash",
            "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential.",
            "genres_list": ["Drama", "Music"],
            "year": 2014,
            "language_str": "English",
        },
    ]

    df = pd.DataFrame(sample)
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["overview"])
    print(f"✅ Sample dataset loaded with {len(df)} movies.")


# Load on startup
load_dataset()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def find_movie(title: str):
    title_lower = title.lower().strip()
    mask = df["title"].str.lower().str.strip() == title_lower
    if mask.any():
        return df[mask].iloc[0], df[mask].index[0]
    mask2 = df["title"].str.lower().str.contains(title_lower, na=False)
    if mask2.any():
        return df[mask2].iloc[0], df[mask2].index[0]
    return None, None


def row_to_dict(row):
    genres = row.get("genres_list", [])
    if not isinstance(genres, list):
        genres = []
    year = int(row.get("year", 0)) if row.get("year") else None
    return {
        "title": row["title"],
        "overview": row["overview"],
        "genres": genres,
        "release_year": year if year and year > 1800 else None,
        "language": row.get("language_str", "Unknown"),
    }


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "CineAI Movie API is running 🎬", "endpoints": ["/search", "/recommend", "/verdict"]}


@app.get("/search")
def search(title: str):
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")
    if not title or len(title.strip()) < 1:
        raise HTTPException(status_code=400, detail="Title parameter is required.")

    row, _ = find_movie(title)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")

    return {"success": True, "data": row_to_dict(row)}


@app.get("/recommend")
def recommend(title: str):
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    row, idx = find_movie(title)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")

    movie_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()
    sim_scores[idx] = -1  # exclude self

    top_indices = np.argsort(sim_scores)[::-1][:5]
    recommendations = []
    for i in top_indices:
        r = df.iloc[i]
        recommendations.append({
            "title": r["title"],
            "genres": r.get("genres_list", []) if isinstance(r.get("genres_list"), list) else [],
            "release_year": int(r["year"]) if r.get("year") and int(r.get("year", 0)) > 1800 else None,
            "similarity_score": round(float(sim_scores[i]), 3),
        })

    return {"success": True, "source_movie": row["title"], "recommendations": recommendations}


@app.get("/verdict")
def verdict(title: str):
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    row, _ = find_movie(title)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")

    score = 0
    reasons = []

    # Criterion 1: Overview depth
    overview_len = len(row["overview"].split())
    if overview_len > 30:
        score += 1
        reasons.append(f"rich narrative description ({overview_len} words)")

    # Criterion 2: Popular / acclaimed genres
    genres_lower = {g.lower() for g in (row.get("genres_list") or [])}
    matched = genres_lower & POPULAR_GENRES
    if matched:
        score += 1
        reasons.append(f"acclaimed genre{'s' if len(matched) > 1 else ''} ({', '.join(matched)})")

    # Criterion 3: Recency
    year = int(row.get("year", 0)) if row.get("year") else 0
    if year >= 2015:
        score += 1
        reasons.append(f"modern release ({year})")

    verdict_text = "Worth Watching" if score >= 2 else "Not Worth Watching"
    reason_text = (
        f"This film scores {score}/3 — featuring {', and '.join(reasons)}."
        if reasons
        else "Insufficient data to make a strong recommendation."
    )

    return {
        "success": True,
        "title": row["title"],
        "verdict": verdict_text,
        "score": score,
        "max_score": 3,
        "reason": reason_text,
    }
