"""
MLVerse X — Redis Client
"""
import redis.asyncio as aioredis
from typing import Optional
from app.core.config import settings

redis_pool: Optional[aioredis.Redis] = None


import asyncio
import logging

logger = logging.getLogger(__name__)

class MockRedisPipeline:
    def __init__(self, mock_redis):
        self.mock_redis = mock_redis
        self.commands = []

    def incr(self, key, amount=1):
        self.commands.append(("incr", key, amount))
        return self

    def expire(self, key, time):
        self.commands.append(("expire", key, time))
        return self

    async def execute(self):
        results = []
        for cmd, *args in self.commands:
            if cmd == "incr":
                key, amount = args
                val = await self.mock_redis.incr(key, amount)
                results.append(val)
            elif cmd == "expire":
                results.append(True)
        return results

class MockRedis:
    def __init__(self):
        self.store = {}

    async def ping(self):
        return True

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int = None):
        self.store[key] = str(value)

    async def delete(self, key: str):
        self.store.pop(key, None)

    async def exists(self, key: str) -> int:
        return 1 if key in self.store else 0

    async def incr(self, key: str, amount: int = 1) -> int:
        val = int(self.store.get(key, 0)) + amount
        self.store[key] = str(val)
        return val

    def pipeline(self):
        return MockRedisPipeline(self)

async def init_redis_pool():
    global redis_pool
    try:
        redis_pool = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        # Timeout quickly if host is offline
        await asyncio.wait_for(redis_pool.ping(), timeout=1.0)
        logger.info("Successfully connected to Redis Cache.")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis at {settings.REDIS_URL} (Error: {e}). Falling back to in-memory MockRedis.")
        redis_pool = MockRedis()


async def get_redis():
    if redis_pool is None:
        await init_redis_pool()
    return redis_pool


class RedisCache:
    """Helper for common Redis cache operations."""

    def __init__(self, prefix: str = "mlverse"):
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[str]:
        r = await get_redis()
        return await r.get(self._key(key))

    async def set(self, key: str, value: str, expire: int = 300) -> None:
        r = await get_redis()
        await r.set(self._key(key), value, ex=expire)

    async def delete(self, key: str) -> None:
        r = await get_redis()
        await r.delete(self._key(key))

    async def exists(self, key: str) -> bool:
        r = await get_redis()
        return bool(await r.exists(self._key(key)))

    async def incr(self, key: str, expire: int = 60) -> int:
        r = await get_redis()
        pipe = r.pipeline()
        await pipe.incr(self._key(key))
        await pipe.expire(self._key(key), expire)
        results = await pipe.execute()
        return results[0]
