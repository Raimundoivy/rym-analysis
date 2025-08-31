import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import hstack
import os

# Import functions from the new train_model.py
from train_model import load_and_clean_data, featurize_data, get_top_genres

# --- Data Loading and Preparation ---
def load_and_clean_data():
    df = pd.read_csv('rym_clean1.csv')
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df.dropna(subset=['avg_rating', 'rating_count', 'review_count', 'primary_genres', 'release_year'], inplace=True)
    df['descriptors'] = df['descriptors'].fillna('')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
    df.dropna(subset=['rating_count', 'review_count'], inplace=True)
    return df

df = load_and_clean_data()
y = df['avg_rating']

# --- Define Model Versions and their artifacts ---
model_configs = {
    "original": {
        "model_path": 'rym_rating_predictor_model.joblib',
        "features_path": 'top_25_genres.joblib', # This is a list of genres, not feature names
        "tfidf_path": None,
        "version_str": "original"
    },
    "v2_RandomForest": {
        "model_path": 'rym_rating_predictor_model_v2.joblib',
        "features_path": 'model_features_v2.joblib',
        "tfidf_path": None,
        "version_str": "v2"
    },
    "v3_RandomForest_Desc": {
        "model_path": 'rym_rating_predictor_model_v3.joblib',
        "features_path": 'model_features_v3.joblib',
        "tfidf_path": 'tfidf_vectorizer_v3.joblib',
        "version_str": "v3"
    },
    "v4_XGBoost_Desc": {
        "model_path": 'rym_rating_predictor_model_v4.joblib',
        "features_path": 'model_features_v4.joblib',
        "tfidf_path": 'tfidf_vectorizer_v4.joblib',
        "version_str": "v4"
    },
    "v5_Tuned_XGBoost_Desc": {
        "model_path": 'rym_rating_predictor_model_v5_tuned.joblib',
        "features_path": 'model_features_v5.joblib',
        "tfidf_path": 'tfidf_vectorizer_v5.joblib',
        "version_str": "v5"
    }
}

results = {}

for name, config in model_configs.items():
    print(f"Evaluating {name}...")
    try:
        model = joblib.load(config["model_path"])
        
        tfidf_vectorizer = None
        if config["tfidf_path"] and os.path.exists(config["tfidf_path"]):
            tfidf_vectorizer = joblib.load(config["tfidf_path"])

        # Prepare features using the featurize_data function
        X_combined, _, _ = featurize_data(df, config["version_str"], tfidf_vectorizer)

        # Split data consistently with how it was trained
        _, X_test, _, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
        
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {"MAE": mae, "MSE": mse, "R2": r2}

    except FileNotFoundError as e:
        print(f"Skipping {name}: Model or artifact file not found ({e}). Please ensure all models are trained.")
    except Exception as e:
        print(f"An error occurred while evaluating {name}: {e}")

# --- Display Results ---
if results:
    results_df = pd.DataFrame(results).T
    print("\n--- Model Performance Comparison ---")
    print(results_df)
else:
    print("No models were successfully evaluated.")