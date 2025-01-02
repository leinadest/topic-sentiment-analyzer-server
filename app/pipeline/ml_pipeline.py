import os
import tarfile

import joblib
import pandas as pd
from google.cloud import storage


def download_pipeline(bucket_name, pipeline_path, local_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(pipeline_path)
    blob.download_to_filename(local_path)

    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall()

    os.remove(local_path)


class Pipeline:
    def __init__(self, pipeline_path):
        self.pipeline = joblib.load(pipeline_path)

    def predict(self, comments):
        input = pd.DataFrame({'text': comments})
        predictions = self.pipeline.predict(input)

        pred_dict = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'scared', 4: 'neutral'}

        outcomes = [pred_dict[pred] for pred in predictions]
        return outcomes
