import os
from enum import Enum
from contextlib import asynccontextmanager
from unittest.mock import patch

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import app.pipeline.preprocessors as prep
from app.config import settings
from app.scraper_service import ScraperService
from app.pipeline.ml_pipeline import Pipeline, download_pipeline

# Load pipeline

pipeline: Pipeline | None = None


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

    # Download pipeline
    download_path = settings.model_path.replace('.tar.gz', '.joblib')
    if not os.path.exists(download_path):
        download_pipeline(
            settings.bucket_name, settings.model_path, settings.model_path
        )

    # Load pipeline
    pipeline = Pipeline(download_path)

    # Start app
    yield

    # Clean up
    os.remove(download_path)


# Set up FastAPI

app = FastAPI(lifespan=lifespan)

origins = [
    settings.client_origin,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
