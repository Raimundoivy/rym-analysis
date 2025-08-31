import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from scipy.sparse import hstack
import joblib
import numpy as np
import argparse
import os

def load_and_clean_data(file_path='rym_clean1.csv'):
    df = pd.read_csv(file_path)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df.dropna(subset=['avg_rating', 'rating_count', 'primary_genres', 'release_year'], inplace=True)
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df.dropna(subset=['rating_count'], inplace=True)
    
    # For models v2-v5, review_count is also used and needs cleaning
    if 'review_count' in df.columns:
        df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
        df.dropna(subset=['review_count'], inplace=True)
    
    # For models v3-v5, descriptors are used and need filling
    if 'descriptors' in df.columns:
        df['descriptors'] = df['descriptors'].fillna('')
        
    return df

def get_top_genres(df, n=25):
    return df['primary_genres'].str.split(', ').explode().value_counts().nlargest(n).index.tolist()

def featurize_data(df, model_version, tfidf_vectorizer=None):
    X_numerical = None
    X_genres = None
    X_tfidf = None
    feature_names = []
    
    top_genres = get_top_genres(df)
    
    # Numerical Features
    if model_version in ['v2', 'v3', 'v4', 'v5']:
        X_numerical = df[['release_year', 'rating_count', 'review_count']].values
        feature_names.extend(['release_year', 'rating_count', 'review_count'])
    else: # Original model
        X_numerical = df[['release_year', 'rating_count']].values
        feature_names.extend(['release_year', 'rating_count'])

    # Genre Features (consistent manual one-hot encoding for v2-v5, original uses MLB)
    genre_features_df = pd.DataFrame()
    for genre in top_genres:
        genre_features_df[f'genre_{genre}'] = df['primary_genres'].str.contains(genre, na=False).astype(int)
    X_genres = genre_features_df.values
    feature_names.extend(genre_features_df.columns.tolist())

    # TF-IDF Features for Descriptors
    if model_version in ['v3', 'v4', 'v5']:
        if tfidf_vectorizer is None:
            tfidf_vectorizer = TfidfVectorizer(max_features=100)
            X_tfidf = tfidf_vectorizer.fit_transform(df['descriptors'])
        else:
            X_tfidf = tfidf_vectorizer.transform(df['descriptors'])
        feature_names = tfidf_vectorizer.get_feature_names_out().tolist() + feature_names
        
    # Combine features
    if model_version in ['v3', 'v4', 'v5']:
        X_combined = hstack([X_tfidf, X_numerical, X_genres])
    elif model_version == 'v2':
        X_combined = np.hstack([X_numerical, X_genres])
    else: # Original model
        X_combined = np.hstack([X_numerical, X_genres]) # Assuming X_genres is already correctly shaped for original
        
    return X_combined, feature_names, tfidf_vectorizer

def train_model(X, y, model_version):
    model = None
    if model_version in ['v2', 'v3']:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
    elif model_version in ['v4']:
        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
    elif model_version == 'v5':
        print("Starting hyperparameter tuning for XGBoost (v5)...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 1.0]
        }
        xgb = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
        grid_search.fit(X, y)
        print(f"Best parameters found: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else: # Original model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
    return model

def save_artifacts(model, feature_names, tfidf_vectorizer, model_version):
    if model_version == 'v5':
        joblib.dump(model, 'rym_rating_predictor_model_v5_tuned.joblib')
        joblib.dump(feature_names, 'model_features_v5.joblib')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_v5.joblib')
        print("Tuned model v5 (XGBoost) and artifacts saved successfully.")
    elif model_version == 'v4':
        joblib.dump(model, 'rym_rating_predictor_model_v4.joblib')
        joblib.dump(feature_names, 'model_features_v4.joblib')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_v4.joblib')
        print("Model v4 (XGBoost) and artifacts saved successfully.")
    elif model_version == 'v3':
        joblib.dump(model, 'rym_rating_predictor_model_v3.joblib')
        joblib.dump(feature_names, 'model_features_v3.joblib')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_v3.joblib')
        print("Model v3 and artifacts saved successfully.")
    elif model_version == 'v2':
        joblib.dump(model, 'rym_rating_predictor_model_v2.joblib')
        joblib.dump(feature_names, 'model_features_v2.joblib')
        print("Model v2 and artifacts saved successfully.")
    else: # Original model
        joblib.dump(model, 'rym_rating_predictor_model.joblib')
        joblib.dump(get_top_genres(load_and_clean_data()), 'top_25_genres.joblib') # Re-save top genres for consistency
        print("Original model and artifacts saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Train various versions of the RYM rating prediction model.")
    parser.add_argument('--version', type=str, default='original', 
                        choices=['original', 'v2', 'v3', 'v4', 'v5'],
                        help="Specify the model version to train (original, v2, v3, v4, v5).")
    args = parser.parse_args()

    print(f"Training model version: {args.version}")

    df = load_and_clean_data()
    y = df['avg_rating']

    # Initialize tfidf_vectorizer outside featurize_data for consistent saving
    tfidf_vectorizer = None
    if args.version in ['v3', 'v4', 'v5']:
        # Fit TF-IDF here to ensure it's fitted on the full data before split
        # and can be saved. featurize_data will then just transform.
        tfidf_vectorizer = TfidfVectorizer(max_features=100)
        tfidf_vectorizer.fit(df['descriptors'])

    X_combined, feature_names, _ = featurize_data(df, args.version, tfidf_vectorizer)

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, args.version)
    save_artifacts(model, feature_names, tfidf_vectorizer, args.version)

if __name__ == "__main__":
    main()