from enum import Enum
from contextlib import asynccontextmanager
from unittest.mock import patch

from fastapi import FastAPI
from pydantic import BaseModel

import app.pipeline.preprocessors as prep
from app.config import settings
from app.scraper_service import ScraperService
from app.pipeline.ml_pipeline import Pipeline

# Load pipeline

pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    # Patch all necessary preprocessors in __main__ before model loading
    patch('__main__.RedditTextCleaner', prep.RedditTextCleaner, create=True).start()
    patch('__main__.Tokenizer', prep.Tokenizer, create=True).start()
    patch('__main__.FeatureEngineer', prep.FeatureEngineer, create=True).start()
    patch('__main__.TfidfTransformer', prep.TfidfTransformer, create=True).start()
    patch('__main__.Scaler', prep.Scaler, create=True).start()
    patch('__main__.Cleaner', prep.Cleaner, create=True).start()

    # Load pipeline
    await pipeline.download(
        settings.bucket_name, settings.model_path, settings.model_path
    )
    pipeline.load()

    # Start app
    yield


# Set up FastAPI

app = FastAPI(lifespan=lifespan)

# Define types


class Time(Enum):
    ALL = "all"
    DAY = "day"
    HOUR = "hour"
    MONTH = "month"
    WEEK = "week"
    YEAR = "year"


class PredictionInput(BaseModel):
    query: str
    time_filter: Time


class PredictionOutput(BaseModel):
    submission_count: int
    comments: list
    predictions: list


# Define routes


@app.post("/predict/")
async def predict(input_data: PredictionInput) -> PredictionOutput:
    global pipeline

    # Get comments
    scraper_service = ScraperService(
        settings.reddit_client_id,
        settings.reddit_client_secret,
        settings.reddit_user_agent,
    )
    async with scraper_service:
        comments, submission_count = await scraper_service.get_comments(
            input_data.query, input_data.time_filter.value
        )

    # Make predictions
    predictions = pipeline.predict(comments)

    return {
        "submission_count": submission_count,
        "comments": comments,
        "predictions": predictions,
    }


@app.get("/health/")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
