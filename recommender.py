import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random


# ================== Load Dataset ==================
def load_dataset(csv_path="data/spotify_songs.csv", sample_size=5000):
    df = pd.read_csv(csv_path)

    # Keep only relevant columns
    features = ["track_name", "artist_name", "valence", "energy", "danceability", "tempo"]
    df = df[features].dropna()

    # Take a sample (to keep it lightweight)
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    return df


# ================== Mood Labeling ==================
def mood_classify(row):
    if row["valence"] > 0.6 and row["energy"] > 0.6:
        return "Happy"
    elif row["valence"] < 0.4 and row["energy"] < 0.4:
        return "Sad"
    elif row["energy"] > 0.7:
        return "Energetic"
    else:
        return "Calm"


def add_mood_labels(df):
    df["mood"] = df.apply(mood_classify, axis=1)
    return df


# ================== Recommender ==================
def recommend_songs(df, mood_label, top_n=5):
    mood_songs = df[df["mood"] == mood_label].reset_index(drop=True)

    if mood_songs.empty:
        return pd.DataFrame(columns=df.columns)

    # Features
    features = mood_songs[["valence", "energy", "danceability", "tempo"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Cosine similarity
    sim = cosine_similarity(X_scaled)

    # Pick a random seed (within reindexed DataFrame)
    seed_idx = random.choice(range(len(mood_songs)))
    similar_idx = sim[seed_idx].argsort()[-top_n:]

    return mood_songs.iloc[similar_idx][["track_name", "artist_name", "mood"]]



# ================== Test Pipeline ==================
if __name__ == "__main__":
    df = load_dataset("data/spotify_songs.csv", sample_size=2000)
    df = add_mood_labels(df)

    print("🎵 Example Mood Counts:")
    print(df["mood"].value_counts())

    print("\n🎵 Recommendations for mood = Happy:")
    print(recommend_songs(df, "Happy", top_n=5))
