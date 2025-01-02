import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables


class Settings(BaseSettings):
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    bucket_name: str
    model_path: str
    google_application_credentials: Optional[str] = None
    client_origin: str

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()

# Set environment variables

if settings.google_application_credentials is not None:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
        settings.google_application_credentials
    )
