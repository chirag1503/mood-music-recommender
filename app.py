import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from recommender import load_dataset, add_mood_labels, recommend_songs

# ================== Streamlit Config ==================
st.set_page_config(page_title="🎵 Mood-Based Music Recommender", layout="centered")

st.title("🎵 Mood-Based Music Recommender")
st.write("Discover new songs based on your mood!")

# ================== Load & Prepare Data ==================
@st.cache_data
def get_data():
    df = load_dataset("data/spotify_songs.csv", sample_size=3000)
    df = add_mood_labels(df)
    return df

df = get_data()

# ================== Mood Distribution Chart ==================
st.subheader("📊 Mood Distribution in Dataset")
mood_counts = df["mood"].value_counts()

fig, ax = plt.subplots()
sns.barplot(x=mood_counts.index, y=mood_counts.values, palette="Set2", ax=ax)
ax.set_ylabel("Number of Songs")
ax.set_xlabel("Mood")
st.pyplot(fig)

st.markdown("---")

# ================== Mood Selection ==================
mood = st.selectbox("Select your mood:", ["Happy", "Sad", "Energetic", "Calm"])

if st.button("Recommend Songs"):
    recs = recommend_songs(df, mood, top_n=5)

    if recs.empty:
        st.error("No songs found for this mood 😢")
    else:
        # Show seed song
        seed_song = recs.iloc[-1]  # last one is the seed in our recommender
        st.success(f"🎵 Recommendations based on **{seed_song['track_name']}** by *{seed_song['artist_name']}*")

        # Show recommendations
        for i, row in recs.iterrows():
            st.write(f"🎶 **{row['track_name']}** by *{row['artist_name']}* ({row['mood']})")
            st.markdown("---")

# ================== Feature Visualization ==================
st.subheader("🎛 Energy vs Valence Distribution")
fig2, ax2 = plt.subplots()
sns.scatterplot(
    data=df.sample(500, random_state=42),  # sample for clarity
    x="valence", y="energy", hue="mood", palette="Set2", ax=ax2
)
ax2.set_title("Energy vs Valence by Mood")
st.pyplot(fig2)
