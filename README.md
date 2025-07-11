# 🎬 Movie Recommendation App

A Streamlit-based web app that recommends movies based on your selected title using item-based collaborative filtering.

### Running the App

**Option 1: Streamlit (Local)**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Option 2: Run with Docker**
```bash
docker pull pavansingu/streamlit-rec-app:latest
docker run -p 8501:8501 pavansingu/streamlit-rec-app:latest
```

### Features

- Recommend top 5 similar movies based on cosine similarity.
- Interactive UI built with Streamlit.
- Movie poster images fetched using DuckDuckGo Search (ddgs).
- Pretrained KNN model for fast recommendations.
- Docker support for easy deployment.

### Working

- Input data from `movies.csv` and `ratings.csv` is used to build a user-item matrix.
- A pivot table is created to represent users' ratings for movies.
- Cosine similarity is calculated between items (movies).
- A K-Nearest Neighbors model (`NearestNeighbors` from `sklearn`) is trained on this matrix.
- When a movie is selected, the app finds and displays 5 most similar movies.
- Poster images are fetched using the `ddgs` (DuckDuckGo Search) package.

### Input Files

- `movies.csv` — Metadata about each movie (e.g., movieId, title).
- `ratings.csv` — User ratings for movies (userId, movieId, rating).
- `knn_recsy_model.pkl`: Pretrained KNN model using cosine similarity.
- A correlation matrix using cosine distance is computed and used during recommendation.
