import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables


class Settings(BaseSettings):
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    bucket_name: str
    model_path: str
    google_application_credentials: str

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()

# Set environment variables

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = settings.google_application_credentials
