import os

import pandas as pd
import pytest

import app.model.preprocessors as prep
from app.model.ml_model import load_model, download_model


@pytest.fixture()
def setup_cleanup(monkeypatch):
    # Set up preprocessors in test environment
    monkeypatch.setattr(
        '__main__.RedditTextCleaner', prep.RedditTextCleaner, raising=False
    )
    monkeypatch.setattr('__main__.Tokenizer', prep.Tokenizer, raising=False)
    monkeypatch.setattr('__main__.FeatureEngineer', prep.FeatureEngineer, raising=False)
    monkeypatch.setattr(
        '__main__.TfidfTransformer', prep.TfidfTransformer, raising=False
    )
    monkeypatch.setattr('__main__.Scaler', prep.Scaler, raising=False)
    monkeypatch.setattr('__main__.Cleaner', prep.Cleaner, raising=False)

    # Set up test environment
    bucket_name = 'leinadest-mlflow-artifacts-store'
    model_path = 'smsa_svc.tar.gz'
    local_path = 'smsa_svc.tar.gz'

    # Run test
    yield bucket_name, model_path, local_path

    # Clean up downloads
    os.remove(local_path)
    os.remove(local_path.replace('.tar.gz', '.joblib'))


@pytest.mark.asyncio
async def test_load_model(setup_cleanup):
    bucket_name, model_path, local_path = setup_cleanup
    test_data = pd.DataFrame({'text': ['I love this movie!']})

    await download_model(bucket_name, model_path, local_path)
    model = load_model(local_path)
    prediction = model.predict(test_data)

    assert prediction[0] == 0
