import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("../data/tmdb_5000_movies.csv")
credits = pd.read_csv("../data/tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, left_on="id", right_on="movie_id", how="left")

# Fix title column after merge
if "title_x" in movies.columns:
    movies.rename(columns={"title_x": "title"}, inplace=True)
if "title_y" in movies.columns:
    movies.drop(columns=["title_y"], inplace=True)

# Keep only needed columns
movies = movies[["title", "overview", "genres", "keywords", "cast", "crew"]]

# Handle missing values
movies.fillna("", inplace=True)

# Helper functions
def convert(obj):
    """Convert list-like JSON string into keywords."""
    try:
        items = [i["name"] for i in ast.literal_eval(obj)]
        return " ".join(items)
    except:
        return ""

def convert_cast(obj):
    """Take top 3 cast members."""
    try:
        items = [i["name"] for i in ast.literal_eval(obj)[:3]]
        return " ".join(items)
    except:
        return ""

def extract_director(obj):
    """Extract director from crew."""
    try:
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return i["name"]
        return ""
    except:
        return ""

# Apply transformations
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(extract_director)

# Create tags column
movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["cast"] + " " +
    movies["crew"]
)

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_title):
    if movie_title not in movies["title"].values:
        print("Movie not found in dataset.")
        return

    index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\nTop recommendations for '{movie_title}':\n")
    for i in movie_list:
        print(movies.iloc[i[0]].title)

# Example usage
recommend("Avatar")
