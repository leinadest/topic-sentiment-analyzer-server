import os
import tarfile

import joblib
import pandas as pd
from gcloud.aio.storage import Storage


class Pipeline:
    def __init__(self):
        self.pipeline = None
        self.pipeline_path = None

    async def download(self, bucket_name, model_path, local_path):
        async with Storage() as storage_client:
            await storage_client.download_to_filename(
                bucket_name, model_path, local_path
            )
        with tarfile.open(local_path, "r:gz") as tar:
            tar.extractall()
        self.pipeline_path = local_path.replace('.tar.gz', '.joblib')
        os.remove(local_path)

    def load(self):
        self.pipeline = joblib.load(self.pipeline_path)
        os.remove(self.pipeline_path)

    def predict(self, comments):
        input = pd.DataFrame({'text': comments})
        predictions = self.pipeline.predict(input)

        pred_dict = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'scared', 4: 'neutral'}

        outcomes = [pred_dict[pred] for pred in predictions]
        return outcomes
