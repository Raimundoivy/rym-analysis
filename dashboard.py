# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Anatomy of Acclaim")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('rym_clean1.csv')
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    return df

df = load_data()

# --- Title and Introduction ---
st.title("Anatomy of Acclaim: Deconstructing Album Success")
st.markdown("""
This dashboard is an interactive exploration of the **Rate Your Music (RYM) Top 5000 Albums** dataset. 
It analyzes the trends, genres, and characteristics of critically acclaimed music from 2000-2022.
""")

# --- Interactive Filtering (Moved to Sidebar) ---
st.sidebar.header("Filter Albums by Genre and Year")

# Get a unique, sorted list of all genres for the filter
all_genres = sorted(df['primary_genres'].dropna().str.split(', ').explode().unique())

# Create the multiselect widget
selected_genres = st.sidebar.multiselect(
    "Select Primary Genres:",
    options=all_genres
)

# Create a slider for the release year
year_range = st.sidebar.slider(
    "Select Release Year Range:",
    min_value=int(df['release_year'].min()),
    max_value=int(df['release_year'].max()),
    value=(int(df['release_year'].min()), int(df['release_year'].max()))
)

# --- Filter the DataFrame based on user selections ---
filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df['release_year'] >= year_range[0]) & (filtered_df['release_year'] <= year_range[1])]
if selected_genres:
    filtered_df = filtered_df[filtered_df['primary_genres'].str.contains('|'.join(selected_genres), na=False)]

st.write(f"Displaying **{len(filtered_df)}** albums based on your selection.")
st.dataframe(filtered_df)

# --- EDA Sections ---
st.header("Exploratory Data Analysis")

if not filtered_df.empty:
    # --- Time-Series Plot ---
    with st.expander("Time-Series: Average Rating and Rating Count Over Years"):
        time_series_df = filtered_df.groupby('release_year').agg({'avg_rating': 'mean', 'rating_count': 'mean'}).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_series_df['release_year'], y=time_series_df['avg_rating'],
                        mode='lines',
                        name='Average Rating'))
        fig.add_trace(go.Scatter(x=time_series_df['release_year'], y=time_series_df['rating_count'],
                        mode='lines',
                        name='Average Rating Count',
                        yaxis='y2'))

        fig.update_layout(
            title_text="Average Rating and Rating Count Over Years",
            yaxis=dict(
                title="Average Rating",
                titlefont=dict(
                    color="#1f77b4"
                ),
                tickfont=dict(
                    color="#1f77b4"
                )
            ),
            yaxis2=dict(
                title="Average Rating Count",
                titlefont=dict(
                    color="#ff7f0e"
                ),
                tickfont=dict(
                    color="#ff7f0e"
                ),
                overlaying="y",
                side="right"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Genre Landscape Chart ---
    with st.expander("Genre Landscape: Top 20 Genres"):
        st.subheader("Top 20 Primary Genres")
        top_genres = df['primary_genres'].str.split(', ').explode().value_counts().nlargest(20)
        fig = px.bar(top_genres, x=top_genres.values, y=top_genres.index, orientation='h',
                     labels={'x': 'Number of Albums', 'y': 'Genre'},
                     title="Top 20 Primary Genres")
        st.plotly_chart(fig, use_container_width=True)

    # --- Genre Evolution Stack Plot ---
    with st.expander("Genre Evolution Over Time"):
        st.subheader("Rise and Fall of Genres")
        top_10_genres = df['primary_genres'].str.split(', ').explode().value_counts().nlargest(10).index
        genre_evolution = df[df['primary_genres'].str.contains('|'.join(top_10_genres), na=False)]
        genre_evolution = genre_evolution[genre_evolution['release_year'] >= 2000]
        
        genre_counts = genre_evolution.groupby(['release_year', 'primary_genres']).size().reset_index(name='count')
        
        genre_counts['primary_genres'] = genre_counts['primary_genres'].str.split(', ')
        genre_counts = genre_counts.explode('primary_genres')
        
        genre_counts = genre_counts[genre_counts['primary_genres'].isin(top_10_genres)]
        
        genre_pivot = genre_counts.groupby(['release_year', 'primary_genres'])['count'].sum().unstack().fillna(0)
        
        st.area_chart(genre_pivot)

    # --- Descriptor Analysis ---
    with st.expander("Descriptor Word Cloud"):
        st.subheader("Most Common Album Descriptors")
        descriptors = filtered_df['descriptors'].dropna().str.cat(sep=', ')
        if descriptors:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(descriptors)
            st.image(wordcloud.to_array(), use_column_width=True)
        else:
            st.warning("No descriptors available for the current selection.")

    # --- Dynamic Genre Deep Dive ---
    st.header("Dynamic Genre Deep Dive")

    # Get a unique, sorted list of all genres for the selection
    all_primary_genres = sorted(df['primary_genres'].dropna().str.split(', ').explode().unique())

    col1, col2 = st.columns(2)
    with col1:
        genre1 = st.selectbox("Select the first genre to compare:", all_primary_genres, index=all_primary_genres.index('Rock'))
    with col2:
        genre2 = st.selectbox("Select the second genre to compare:", all_primary_genres, index=all_primary_genres.index('Hip Hop'))

    tab1, tab2, tab3 = st.tabs(["Rating Comparison", "Top Artists", "Common Descriptors"])

    with tab1:
        st.subheader(f"Rating Distribution: {genre1} vs. {genre2}")
        
        genre1_df = df[df['primary_genres'].str.contains(genre1, na=False) & 
                       (df['release_year'] >= year_range[0]) & 
                       (df['release_year'] <= year_range[1])]
                       
        genre2_df = df[df['primary_genres'].str.contains(genre2, na=False) & 
                       (df['release_year'] >= year_range[0]) & 
                       (df['release_year'] <= year_range[1])]

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=genre1_df['avg_rating'], name=genre1, opacity=0.75))
        fig.add_trace(go.Histogram(x=genre2_df['avg_rating'], name=genre2, opacity=0.75))

        fig.update_layout(barmode='overlay',
                          xaxis_title_text='Average Rating',
                          yaxis_title_text='Count')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Top 10 Artists: {genre1} vs. {genre2}")

        filtered_artists_df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]

        st.markdown(f"**Top 10 {genre1} Artists**")
        genre1_artists = filtered_artists_df[filtered_artists_df['primary_genres'].str.contains(genre1, na=False)]
        st.write(genre1_artists['artist_name'].value_counts().nlargest(10))

        st.markdown(f"**Top 10 {genre2} Artists**")
        genre2_artists = filtered_artists_df[filtered_artists_df['primary_genres'].str.contains(genre2, na=False)]
        st.write(genre2_artists['artist_name'].value_counts().nlargest(10))

    with tab3:
        st.subheader(f"Common Descriptors: {genre1} vs. {genre2}")

        filtered_descriptors_df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]

        st.markdown(f"**Top 10 {genre1} Descriptors**")
        genre1_descriptors = filtered_descriptors_df[filtered_descriptors_df['primary_genres'].str.contains(genre1, na=False)]
        st.write(genre1_descriptors['descriptors'].str.split(', ').explode().value_counts().nlargest(10))

        st.markdown(f"**Top 10 {genre2} Descriptors**")
        genre2_descriptors = filtered_descriptors_df[filtered_descriptors_df['primary_genres'].str.contains(genre2, na=False)]
        st.write(genre2_descriptors['descriptors'].str.split(', ').explode().value_counts().nlargest(10))

else:
    st.warning("No albums match your current filter selection.")

from scipy.sparse import hstack

# --- Rating Predictor Engine (v5 with Tuned XGBoost) ---
st.header("Predict an Album's Rating (v5 Model - Tuned XGBoost)")

# Load the trained model, features, and vectorizer
try:
    model = joblib.load('rym_rating_predictor_model_v5_tuned.joblib')
    model_features = joblib.load('model_features_v5.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer_v5.joblib')
except FileNotFoundError:
    st.error("Model files (v5) not found. Please run `python train_model_v5_tuned.py` first.")
    st.stop()

# Extract genre features from model_features
genre_cols = [f for f in model_features if f.startswith('genre_')]
top_25_genres_for_selectbox = [f.replace('genre_', '') for f in genre_cols]

# Input widgets
col1, col2, col3 = st.columns(3)
with col1:
    predict_release_year = st.slider("Release Year", min_value=1950, max_value=2025, value=2020)
with col2:
    predict_rating_count = st.number_input("Rating Count", min_value=0, value=1000)
with col3:
    predict_review_count = st.number_input("Review Count", min_value=0, value=50)

predict_primary_genre = st.selectbox("Primary Genre", options=top_25_genres_for_selectbox)
predict_descriptors = st.text_input("Descriptors (comma-separated)", "energetic, melodic, male vocals")

if st.button("Predict Acclaim"):
    # 1. TF-IDF for Descriptors
    X_tfidf_pred = tfidf_vectorizer.transform([predict_descriptors])

    # 2. Numerical Features
    X_numerical_pred = np.array([[predict_release_year, predict_rating_count, predict_review_count]])

    # 3. One-Hot Encode Genre
    genre_features_pred = pd.DataFrame(np.zeros((1, len(top_25_genres_for_selectbox))), columns=top_25_genres_for_selectbox)
    if predict_primary_genre in genre_features_pred.columns:
        genre_features_pred[predict_primary_genre] = 1
    X_genres_pred = genre_features_pred.values

    # Combine all features in the correct order
    X_pred_combined = hstack([X_tfidf_pred, X_numerical_pred, X_genres_pred])

    # Predict
    prediction = model.predict(X_pred_combined)
    
    st.success(f"Predicted Average Rating: {prediction[0]:.2f}")


# --- Artist-Specific Analysis ---
st.header("Artist-Specific Analysis")

# Get a unique, sorted list of all artists for the filter
all_artists = sorted(df['artist_name'].dropna().unique())

selected_artist = st.selectbox(
    "Select an Artist:",
    options=all_artists
)

if selected_artist:
    artist_df = df[df['artist_name'] == selected_artist].sort_values('release_date')

    st.subheader(f"Album Performance for {selected_artist}")

    if not artist_df.empty:
        # --- Time-Series Plot for Artist ---
        fig = px.line(artist_df, x='release_date', y='avg_rating', title=f'Album Ratings Over Time for {selected_artist}',
                      labels={'release_date': 'Release Date', 'avg_rating': 'Average Rating'}, markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # --- Display Artist's Albums ---
        st.subheader("Albums")
        st.dataframe(artist_df[['release_name', 'release_year', 'avg_rating', 'rating_count', 'primary_genres']])
    else:
        st.warning("No data available for the selected artist.")