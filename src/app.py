import streamlit as st
import requests

TMDB_API_KEY = "91c81fa8f0f7d30899917353194869e9"


# -----------------------------
# Helper – Fetch movie list for dropdown
# -----------------------------
def load_popular_movies():
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page=1"
    data = requests.get(url).json()
    return [movie["title"] for movie in data["results"]]


# -----------------------------
# Helper – Get poster
# -----------------------------
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    data = requests.get(url).json()
    poster_path = data.get("poster_path")
    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path
    return "https://via.placeholder.com/500x750?text=No+Image"


# -----------------------------
# TMDB Built-In "Similar Movies" API
# -----------------------------
def recommend(movie_title):
    # Step 1: Search movie ID
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    search_data = requests.get(search_url).json()

    if not search_data["results"]:
        return [], []

    movie_id = search_data["results"][0]["id"]

    # Step 2: Fetch similar movies
    similar_url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={TMDB_API_KEY}&language=en-US&page=1"
    similar_data = requests.get(similar_url).json()

    titles = []
    posters = []

    for result in similar_data["results"][:5]:
        titles.append(result["title"])
        posters.append(fetch_poster(result["id"]))

    return titles, posters


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Movie Recommendation System")

movie_list = load_popular_movies()
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.image(posters[idx], use_container_width=True)
            st.text(names[idx])
