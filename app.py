from fastapi import FastAPI
import pandas as pd

app = FastAPI(
    title="LLM Recommendation Engine"
)

# Load recommendation results
recommendations_df = pd.read_csv(
    "data/final_recommendations.csv"
)


@app.get("/")
def home():

    return {
        "message": "Recommendation Engine API Running"
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int):

    recommendations = (
        recommendations_df
        .head(10)
        [
            [
                "title",
                "category",
                "rank_score",
                "explanation"
            ]
        ]
        .to_dict(orient="records")
    )

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }