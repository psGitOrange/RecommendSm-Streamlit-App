import streamlit as st
import pandas as pd
import joblib
# import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# import requests
from ddgs import DDGS
# from PIL import Image
# from io import BytesIO


def extract_title_and_year(title_str):
    match = re.search(r"\((\d{4})\)$", title_str)
    if match:
        year = match.group(1)
        title = re.sub(r"\(\d{4}\)\s*$", "", title_str)
        return pd.Series([title.strip(), int(year)])
    else:
        return pd.Series([title_str.strip(), None])


@st.cache_data
def load_dfs():
    movies_df_raw = pd.read_csv("movies.csv")
    ratings_df_raw = pd.read_csv("ratings.csv")

    selected_movies = ratings_df_raw.movieId.value_counts().head(1000).index.to_list()  # top 1000 movies rated
    movies_df_raw[['title', 'year']] = movies_df_raw['title'].apply(lambda x: extract_title_and_year(x))
    top_movies = movies_df_raw.loc[movies_df_raw.movieId.isin(selected_movies[:50])]

    selected_ratings = ratings_df_raw.loc[ratings_df_raw.movieId.isin(selected_movies)]  # 63k rows data
    ratings_matrix = pd.pivot(data=selected_ratings, index="userId", columns='movieId', values='rating').fillna(0)
    return movies_df_raw, top_movies, ratings_matrix

movies_df_raw, top_movies, ratings_matrix = load_dfs()

# with open('ratings_matrix_movies.pkl', 'wb') as f:
#     pickle.dump(ratings_matrix, f)

# with open('ratings_matrix_movies.pkl', 'rb') as f:
#     ratings_matrix = pickle.load(f)
# print(ratings_matrix.head(4))

# Load model
model_knn = joblib.load('knn_recsy_model.pkl')

def get_similar_movies_knn(movie_id):
    if movie_id not in ratings_matrix.T.index:
        # return f"Movie ID {movie_id} not found in the rating matrix."
        return None, None

    movie_vector = ratings_matrix.T.loc[[movie_id]]  # keep 2D shape
    distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=6)

    similar_titles = movies_df_raw[movies_df_raw['movieId'].isin(indices.flatten())][['movieId', 'title']]
    similarity_scores = 1 - distances.flatten()[1:]  # convert cosine distance to similarity

    return similar_titles, similarity_scores


# load cosine_sim matrix
item_cosine_sim = cosine_similarity(ratings_matrix.T)
item_cosineSim_df = pd.DataFrame(item_cosine_sim,
                                 index=ratings_matrix.columns,
                                 columns=ratings_matrix.columns)


def get_top5_similar_movies(movie_id):
    if movie_id not in item_cosineSim_df.columns:
        # return f"Movie ID {movie_id} not found in the correlation matrix."
        return None, None

    # Drop self-correlation and sort values
    similar_movies = item_cosineSim_df[movie_id].drop(labels=[movie_id]).sort_values(ascending=False).head(5)
    # top_similar = similar_movies.sort_values(ascending=False).head(5)
    similar_titles = movies_df_raw[movies_df_raw['movieId'].isin(similar_movies.index)][['movieId', 'title']]

    return similar_titles, similar_movies.values


ddgs = DDGS()
@st.cache_data(show_spinner=False)
def get_poster_url(title):
    results = ddgs.images(keywords=f"{title} movie poster", max_results=1)
    if results:
        return results[0]['image']
    return None


# ---------app starts here---------
st.markdown("<style>.block-container {max-width: 800px; margin: auto;}</style>",
            unsafe_allow_html=True)
st.title('Movie Recommendation')

selected_movie_title = st.selectbox("Select a movie", top_movies['title'].sort_values())
print("selected_movie_title:", selected_movie_title)

if st.button("Recommend"):
    movie_id = top_movies[top_movies['title']==selected_movie_title]['movieId'].values[0]
    print("movie_id:", movie_id)

    # knn_rec_df, scores = get_similar_movies_knn(movie_id)
    # print("knn_rec_titles:", knn_rec_df)
    # print("scores:",scores)
    # rec_movies = knn_rec_df['title'].to_list()

    cosine_rec_df, scores = get_top5_similar_movies(movie_id)
    print("cosine_rec_df:", cosine_rec_df)
    print("scores:",scores)
    rec_movies = cosine_rec_df['title'].to_list()

    if rec_movies:
        st.subheader("Top 5 Recommendations:")
        cols, html_blocks = st.columns(5), []
        for i, title in enumerate(rec_movies):
            col = cols[i % 5]
            with col:
                st.markdown(f"**{i+1}. {title}**")
                try:
                    poster_url = get_poster_url(title)
                    st.image(poster_url, width=120)
                except Exception as e:
                    st.write(f"No poster found. Exception occured as: {e}")

    else:
        st.write("No recommendations found for this movie.")