# **Movie Recommendation System**

This project is a content-based movie recommendation system built using cosine similarity and metadata from the TMDB 5000 Movies dataset. It allows users to select a movie and receive ranked recommendations based on similarity in genres, keywords, cast, and crew roles.

The project also includes an interactive Streamlit interface that displays movie posters fetched from the TMDB API.

---

## **Features**

* Content-based recommendations using cosine similarity
* Metadata processing for genres, keywords, cast, and crew
* Automated poster retrieval using the TMDB API
* Interactive Streamlit user interface
* Clean and modular code structure
* Works without training any machine-learning model

---

## **Project Structure**

```
Movie-Recommendation-System/
│
├── data/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   └── similarity.pkl        (ignored in Git, generated locally)
│
├── src/
│   ├── preprocess.py         (data cleaning and preprocessing)
│   ├── recommend.py          (similarity model and recommendation logic)
│   └── app.py                (Streamlit application)
│
├── requirements.txt
└── README.md
```

---

## **How It Works**

1. The raw TMDB datasets are loaded and merged
2. Text-based fields (genres, keywords, cast, crew) are cleaned and standardized
3. A combined “tags” column is created for each movie
4. A TF-IDF-like vectorization converts text into feature vectors
5. Cosine similarity is computed between all movie vectors
6. Recommendations are based on the closest similarity scores
7. Movie posters are retrieved from TMDB API for visualization in Streamlit

The similarity matrix (`similarity.pkl`) is generated once and reused to speed up recommendations.

---

## **Running the Project**

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit app:

```
streamlit run src/app.py
```

This will open the interface in your browser at:

```
http://localhost:8501
```

Select a movie from the dropdown to view recommendations with posters.

---

## **Datasets**

The project uses the publicly available TMDB 5000 Movies dataset:

* `tmdb_5000_movies.csv`
* `tmdb_5000_credits.csv`

These files contain movie-level metadata such as:

* Genres
* Keywords
* Cast & crew
* Revenue, ratings, runtime
* Overviews

---

## **Requirements**

All required Python packages are listed in:

```
requirements.txt
```

Typical dependencies include:

* pandas
* numpy
* scikit-learn
* streamlit
* requests

---

## **Future Improvements**

* Add collaborative filtering (user–item matrix factorization)
* Hybrid recommendation engine combining content-based and CF
* Deploy the Streamlit app online for public access
* Improve metadata extraction for cast/crew roles
* Add a search bar with fuzzy matching

---

## **Author**

Rizwan Ali Mondal