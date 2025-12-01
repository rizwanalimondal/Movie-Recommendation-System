import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv("../data/tmdb_5000_movies.csv")
credits = pd.read_csv("../data/tmdb_5000_credits.csv")

movies = movies.merge(credits, left_on="id", right_on="movie_id", how="left")

if "title_x" in movies.columns:
    movies.rename(columns={"title_x": "title"}, inplace=True)
if "title_y" in movies.columns:
    movies.drop(columns=["title_y"], inplace=True)

movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies.fillna("", inplace=True)

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

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

similarity = cosine_similarity(vectors)

# -----------------------------
# Save preprocessed files
# -----------------------------
pickle.dump(movies, open("../data/movies.pkl", "wb"))
pickle.dump(similarity, open("../data/similarity.pkl", "wb"))

print("Preprocessing complete. Files saved:")
print("movies.pkl")
print("similarity.pkl")
