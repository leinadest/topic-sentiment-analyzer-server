import asyncpraw


class ScraperService:
    def __init__(self, CLIENT_ID, CLIENT_SECRET, USER_AGENT):
        self.reddit = asyncpraw.Reddit(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT
        )

    async def get_submissions(self, query, time_filter):
        query = query.lower()

        subreddit = await self.reddit.subreddit('all')
        submissions_listing = subreddit.search(
            query=query, sort='top', time_filter=time_filter
        )

        return submissions_listing

    async def get_comments(self, query, time_filter):
        query = query.lower()

        submissions_listing = await self.get_submissions(query, time_filter)

        comments = []
        count = 0

        async for submission in submissions_listing:
            count += 1
            await submission.load()
            await submission.comments.replace_more(limit=0)
            for comment in submission.comments:
                if query in comment.body.lower():
                    comments.append(comment.body)
            if count >= 20:
                break

        return comments, count

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.reddit.close()
