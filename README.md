# LLM Powered Hybrid Movie Recommendation System

This project implements a **hybrid movie recommendation system** using collaborative filtering, content-based filtering, and natural language explanations. It is designed to deliver personalized movie suggestions along with user-friendly explanations powered by a local large language model.

##  Features

* **Collaborative Filtering:** Built using the Surprise library and Singular Value Decomposition (SVD).
* **Content-Based Filtering:** Utilizes TF-IDF vectorization on movie genres and tags.
* **Hybrid Recommender:** Combines scores from collaborative and content-based approaches.
* **Cold Start Handling:** Works even when limited user history is available.
* **User Profiling:** Extracts preferred genres and tags from liked movies.
* **Explainable AI:** Generates short natural-language explanations using Mistral 7B LLM locally.

##  Technologies Used

* Python
* Pandas, NumPy
* scikit-surprise
* Scikit-learn
* Matplotlib
* Transformers (`ctransformers`)
* Mistral 7B model (`TheBloke/Mistral-7B-Instruct-v0.1`)
* Google Colab for rapid prototyping

##  Dataset

The model uses the **MovieLens dataset**, specifically:

* `movies.csv`
* `ratings.csv`
* `tags.csv`
* `genome-tags.csv`
* `genome-scores.csv`

Make sure to extract these files into an `RS/` folder:

```
RS/
├── movies.csv
├── ratings.csv
├── tags.csv
├── genome-tags.csv
├── genome-scores.csv
```

> You can download the dataset from [MovieLens website](https://grouplens.org/datasets/movielens/).

##  How It Works

### 1. Data Loading and Preprocessing

* Reads movie metadata, user ratings, and tags.
* Combines tags and genres into unified movie representations.

### 2. Collaborative Filtering (SVD)

* Trains an SVD model on user-item rating matrix.
* Evaluates performance using RMSE and precision/recall at top-K.

### 3. Content-Based Filtering

* Applies TF-IDF vectorization to movie genres and tags.
* Computes cosine similarity to recommend similar movies.

### 4. Hybrid Recommendation

* Merges collaborative and content scores to produce a ranked list.
* Filters out movies already rated by the user.

### 5. Cold Start & Profiling

* Builds a user profile based on a few liked movies.
* Recommends similar titles using tag/genre similarity.

### 6. Natural Language Explanations

* Loads Mistral 7B model locally via `ctransformers`.
* Generates short explanations for each recommendation.

```python
prompt = f"[INST] Explain why the movie '{title}' might be recommended to a user based on their preferences. [/INST]"
```

##  Evaluation Metrics

* **RMSE (Root Mean Squared Error)** for SVD model
* **Precision and Recall** for hybrid recommender
* **Bar plots** to visualize evaluation metrics

##  Output Files

* `recommendations_user1.csv`: Top movie suggestions
* `recommendations_with_explanations.csv`: Final output with AI-generated reasoning

##  Installation

You can run the notebook in Google Colab with the following dependencies:

```bash
pip install numpy==1.24.4 scikit-surprise openpyxl
pip install ctransformers pandas transformers accelerate bitsandbytes
```
