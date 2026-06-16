# Project-LLM-Powered-Recommendation-Engine

# LLM-Powered Personalized Recommendation Engine

An end-to-end recommendation system inspired by Netflix and Perplexity that combines semantic retrieval, learning-to-rank, explainability, and API deployment.

Built using **Sentence Transformers**, **FAISS**, **LightGBM LambdaRank**, and **FastAPI**.

---

## Overview

This project implements a modern multi-stage recommendation pipeline:

1. User interaction modeling
2. Semantic embedding generation
3. FAISS candidate retrieval
4. Feature engineering
5. Learning-to-rank with LightGBM LambdaRank
6. Evaluation metrics
7. Explainable recommendations
8. FastAPI deployment

The system generates personalized recommendations based on:

* Click history
* Search activity
* Watch history
* Likes and engagement patterns

---

##  Architecture

```text
User Interactions
(clicks, likes, searches, watches)
            ↓
Sentence Transformers
            ↓
User Embeddings
            ↓
FAISS Vector Search
            ↓
Candidate Retrieval
            ↓
Feature Engineering
            ↓
LightGBM LambdaRank
            ↓
Top-K Recommendations
            ↓
Explanation Layer
            ↓
FastAPI Deployment
```

---

# 🛠 Tech Stack

### Machine Learning

* Python
* Sentence Transformers
* FAISS
* LightGBM
* Scikit-Learn

### Data Processing

* Pandas
* NumPy

### Deployment

* FastAPI
* Uvicorn

### Storage

* CSV files (can be upgraded to PostgreSQL)

---

# Project Structure

```text
LLM-Powered-Recommendation-Engine/

│
├── data/
│   ├── item_df.csv
│   ├── interactions.csv
│   ├── ranking_results.csv
│   ├── final_recommendations.csv
│   ├── item_embeddings.pkl
│   └── user_embeddings.pkl
│
├── Notebook 1 - Data_Generation.ipynb
├── Notebook 2 - Item Embeddings.ipynb
├── Notebook 3 - User Modeling.ipynb
├── Notebook 4 - FAISS Candidate Retrieval.ipynb
├── Notebook 5 - Feature Engineering.ipynb
├── Notebook 6 - LightGBM LambdaRank.ipynb
├── Notebook 7 - Evaluation Metrics.ipynb
├── Notebook 8 - OpenAI Explanation Layer.ipynb
├── Notebook 9 - FastAPI Deployment.ipynb
│
├── app.py
└── README.md
```

---

# Notebook Workflow

---

## Notebook 1 — Data Generation

Generated synthetic datasets:

### Users

* User IDs

### Items

* Content title
* Category

### Interactions

* Clicks
* Watches
* Likes
* Searches

Outputs:

```text
item_df.csv
interactions.csv
```

---

## Notebook 2 — Item Embeddings

Used Sentence Transformers to generate semantic embeddings for all content items.

Model:

```python
all-MiniLM-L6-v2
```

Outputs:

```text
item_embeddings.pkl
```

---

## Notebook 3 — User Modeling

Built user representations from historical interactions:

* Click history
* Likes
* Watch history

Generated user embeddings.

Outputs:

```text
user_embeddings.pkl
```

---

## Notebook 4 — FAISS Candidate Retrieval

Implemented vector search using FAISS.

Process:

1. Encode user
2. Retrieve nearest items
3. Generate Top-K candidates

---

## Notebook 5 — Feature Engineering

Created ranking features:

* Similarity score
* Popularity score
* Recency score
* User preference score

Generated training dataset for ranking.

---

## Notebook 6 — LightGBM LambdaRank

Trained a learning-to-rank model using:

```python
LGBMRanker
```

Generated:

```text
rank_score
```

for each recommendation candidate.

Outputs:

```text
ranking_results.csv
```

---

## Notebook 7 — Evaluation Metrics

Evaluated recommendation quality using:

### Precision@10

Measures relevance among top recommendations.

### Recall@10

Measures coverage of relevant items.

### MAP@K

Mean Average Precision.

### NDCG@10

Measures ranking quality.

---

## Notebook 8 — Explainable Recommendation Layer

Generated personalized explanations:

Example:

```text
Recommended because you frequently engage with AI-related content.
```

Added interpretability to recommendations.

Outputs:

```text
final_recommendations.csv
```

---

## Notebook 9 — FastAPI Deployment

Converted the recommendation system into a REST API.

Endpoints:

### Home

```text
GET /
```

Response:

```json
{
  "message": "Recommendation Engine API Running"
}
```

### Recommendations

```text
GET /recommend/{user_id}
```

Returns Top-K recommendations with explanations.

---

#  Running the Project

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/LLM-Powered-Recommendation-Engine.git
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Start FastAPI

```bash
uvicorn app:app --reload
```

---

## 4. Open API

Home:

```text
http://127.0.0.1:8000
```

Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

---

# Evaluation Metrics

The project evaluates:

* Precision@10
* Recall@10
* MAP@K
* NDCG@10

These metrics are commonly used in production recommendation systems.

---

# Key Features

✅ Semantic Retrieval using Sentence Transformers

✅ FAISS Vector Search

✅ Multi-Stage Recommendation Pipeline

✅ Learning-to-Rank using LightGBM LambdaRank

✅ Explainable Recommendations

✅ FastAPI Deployment

✅ Interactive Swagger Documentation

---

# Future Improvements

* PostgreSQL integration
* Redis caching
* Docker support
* CI/CD pipelines
* MLflow experiment tracking
* Real datasets (MovieLens, Amazon Reviews)
* Online inference pipeline

---

# Resume Highlights

* Built an end-to-end recommendation engine using Sentence Transformers and FAISS for semantic retrieval.
* Designed user embeddings from clicks, likes, searches, and watch history.
* Implemented LightGBM LambdaRank for Top-K ranking.
* Evaluated ranking quality using Precision@10, Recall@10, MAP, and NDCG.
* Added explainable recommendation generation.
* Deployed the system using FastAPI and Swagger.

---

## Author

**Amit Yadav**

Machine Learning Engineer | Generative AI | Recommendation Systems
