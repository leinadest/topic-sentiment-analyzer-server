import tarfile

import joblib
from gcloud.aio.storage import Storage

from app.model.preprocessors import (  # noqa: F401; Needed for joblib.load
    Scaler,
    Cleaner,
    Tokenizer,
    FeatureEngineer,
    TfidfTransformer,
    RedditTextCleaner,
)


async def download_model(bucket_name, model_path, local_path):
    async with Storage() as storage_client:
        await storage_client.download_to_filename(bucket_name, model_path, local_path)


def load_model(local_path):
    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall()

    model = joblib.load(local_path.replace('.tar.gz', '.joblib'))
    return model
