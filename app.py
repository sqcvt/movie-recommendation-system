import numpy as np
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests 

# ===== DATA PROCESSING =====
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['genres']   = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast']     = movies['cast'].apply(convert3)
movies['crew']     = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['cast']     = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['genres']   = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew']     = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df = movies[['id','title','tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x)).str.lower()

ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(i) for i in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

@st.cache_data
def get_similarity():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    return cosine_similarity(vectors)

similarity = get_similarity()

# ===== RECOMMEND FUNCTION =====
def recommend(movie):
    idx = new_df[new_df['title'] == movie].index[0]
    distances = similarity[idx]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]

# ===== STREAMLIT UI =====
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")
st.markdown("Find movies similar to your favorites!")

selected_movie = st.selectbox("Select a movie:", new_df['title'].values)

if st.button(" Recommendations"):
    recommendations = recommend(selected_movie)
    st.subheader(f"Movies similar to **{selected_movie}**:")
    cols = st.columns(5)
    for idx, movie in enumerate(recommendations):
        with cols[idx]:
            st.markdown(f"""
            <div style='background-color:#1e1e2e;padding:15px;border-radius:10px;text-align:center;'>
                <p style='font-size:40px'>🎬</p>
                <p style='color:white;font-weight:bold;font-size:13px'>{movie}</p>
            </div>
            """, unsafe_allow_html=True)