import os

import pytest

import app.pipeline.preprocessors as prep
from app.pipeline.ml_pipeline import Pipeline, download_pipeline


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

    # Clean up test environment
    os.remove(local_path.replace('tar.gz', 'joblib'))


@pytest.mark.asyncio
async def test_load_model(setup_cleanup):
    bucket_name, model_path, local_path = setup_cleanup
    X_test = ['I love this movie!']
    y_test = ['happy']

    download_pipeline(bucket_name, model_path, local_path)
    pipeline_path = local_path.replace('tar.gz', 'joblib')
    pipeline = Pipeline(pipeline_path)

    y_pred = pipeline.predict(X_test)

    assert y_pred == y_test
