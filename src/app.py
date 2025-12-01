import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# ENTER YOUR TMDB API KEY HERE
# -----------------------------
TMDB_API_KEY = "91c81fa8f0f7d30899917353194869e9"

# -----------------------------
# Load datasets
# -----------------------------
movies = pd.read_csv("../data/tmdb_5000_movies.csv")
credits = pd.read_csv("../data/tmdb_5000_credits.csv")

movies = movies.merge(credits, left_on="id", right_on="movie_id", how="left")

# Fix title columns after merge
if "title_x" in movies.columns:
    movies.rename(columns={"title_x": "title"}, inplace=True)
if "title_y" in movies.columns:
    movies.drop(columns=["title_y"], inplace=True)

movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies.fillna("", inplace=True)

# -----------------------------
# Preprocessing functions
# -----------------------------
def parse_list(obj):
    try:
        return [i["name"] for i in ast.literal_eval(obj)]
    except:
        return []

def convert(obj):
    return " ".join(parse_list(obj))

def convert_cast(obj):
    return " ".join(parse_list(obj)[:3])

def extract_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return i["name"]
    except:
        pass
    return ""

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(extract_director)

movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["cast"] + " " +
    movies["crew"]
)

# -----------------------------
# Vectorize and compute similarity
# -----------------------------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

similarity = cosine_similarity(vectors)

# -----------------------------
# Fetch movie poster from TMDB API
# -----------------------------
def fetch_poster(movie_title):
    search_url = (
        f"https://api.themoviedb.org/3/search/movie"
        f"?api_key={TMDB_API_KEY}&query={movie_title}"
    )
    
    response = requests.get(search_url)
    data = response.json()

    if data.get("results"):
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    return "https://via.placeholder.com/500x750?text=No+Image"

# -----------------------------
# Recommendation logic
# -----------------------------
def recommend(movie_title):
    index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    titles = []
    posters = []

    for i in movie_list:
        title = movies.iloc[i[0]].title
        titles.append(title)
        posters.append(fetch_poster(title))

    return titles, posters

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie:",
    movies["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for idx, col in enumerate(cols):
        with col:
            st.image(posters[idx], use_container_width=True)
            st.text(names[idx])
