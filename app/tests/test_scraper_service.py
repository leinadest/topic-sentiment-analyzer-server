import pytest

from app.config import *
from app.scraper_service import ScraperService


@pytest.mark.asyncio
async def test_get_comments():
    query = 'apple'
    time = 'week'

    scraper_service = ScraperService(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    comments, submission_count = await scraper_service.get_comments(query, time)
    await scraper_service.close()

    assert submission_count > 0
    assert len(comments) > 0
    assert all(query in comment.lower() for comment in comments)
