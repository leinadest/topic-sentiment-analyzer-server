import pytest

from app.config import settings
from app.scraper_service import ScraperService


@pytest.mark.asyncio
async def test_get_comments():
    query = 'a'
    time = 'all'

    scraper_service = ScraperService(
        settings.reddit_client_id,
        settings.reddit_client_secret,
        settings.reddit_user_agent,
    )
    async with scraper_service:
        comments, submission_count = await scraper_service.get_comments(query, time)

    assert submission_count > 0
    assert len(comments) > 0
    assert all(query in comment.lower() for comment in comments)
