# Anatomy of Acclaim: Deconstructing Album Success (2000-2025)

**A comprehensive analysis and prediction project for music album ratings from Rate Your Music (RYM).**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://shields.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

## Project Description

This project scrapes, analyzes, and models data from Rate Your Music (RYM) to understand the characteristics of critically acclaimed albums from 2000 to 2025. It features an interactive Streamlit dashboard for data exploration and a machine learning model to predict album ratings.

## Key Features

*   **Data Scraping:** A Selenium-based scraper to collect album data from RYM.
*   **Interactive Dashboard:** A Streamlit application for visualizing and filtering the album data.
*   **In-depth Data Analysis:** Explore trends in genres, ratings, and popularity over time.
*   **Rating Prediction:** A machine learning model to predict album ratings based on various features.
*   **Model Evaluation:** Scripts to evaluate and compare the performance of different models.
*   **Containerized Deployment:** A Dockerfile for easy setup and deployment of the application.

## File Descriptions

| File                          | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `dashboard.py`                | The main Streamlit application for the interactive dashboard.               |
| `rym_scraper.py`              | Scrapes album data from Rate Your Music.                                    |
| `train_model.py`              | Trains various versions of the machine learning model.                      |
| `evaluate_models.py`          | Compares the performance of all trained models.                             |
| `rym_clean1.csv`              | The cleaned dataset used for analysis and modeling.                         |
| `rym_top_albums_2000-2025_raw.csv` | The raw, scraped data from RYM.                                         |
| `Dockerfile`                  | For building the Docker container for the application.                      |
| `requirements.txt`            | A list of all the Python packages required to run the project.              |
| `*.joblib`                    | Saved machine learning models, features, and vectorizers.                   |

## Models

This project includes several machine learning models, which can be trained using `train_model.py` with the `--version` flag:

*   **Original:** A RandomForest model using basic numerical and genre features.
*   **v2_RandomForest:** A RandomForest model using basic numerical and genre features, including review count.
*   **v3_RandomForest_Desc:** A RandomForest model that adds TF-IDF vectorized album descriptors.
*   **v4_XGBoost_Desc:** An XGBoost model with the same features as v3.
*   **v5_Tuned_XGBoost_Desc:** The final, hyperparameter-tuned XGBoost model, which is used in the dashboard.

## Installation

You can set up the project in two ways:

### 1. Local Setup

**Prerequisites:**
*   Python 3.11
*   Conda (Recommended)

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rym-analysis
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n rym-analysis python=3.11
    conda activate rym-analysis
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Docker Setup

**Prerequisites:**
*   Docker

**Instructions:**

1.  **Build the Docker image:**
    ```bash
    docker build -t rym-analysis .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 rym-analysis
    ```
    The dashboard will be available at `http://localhost:8501`.

## Usage

1.  **Scrape Data (Optional):**
    To re-scrape the data from Rate Your Music, run:
    ```bash
    python rym_scraper.py
    ```

2.  **Train the Model:**
    To train a specific model version (e.g., v5), run:
    ```bash
    python train_model.py --version v5
    ```
    Available versions: `original`, `v2`, `v3`, `v4`, `v5`.

3.  **Run the Dashboard:**
    If you are not using Docker, run the following command to start the Streamlit dashboard:
    ```bash
    streamlit run dashboard.py
    ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

*   Data sourced from [Rate Your Music](https://rateyourmusic.com/).
*   Built with [Streamlit](https://streamlit.io/).