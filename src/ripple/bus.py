"""Thin wrapper over Redis Streams: publish models, subscribe with consumer groups."""

from collections.abc import Iterator
from typing import TypeVar

import redis
from pydantic import BaseModel

from ripple.config import REDIS_URL

M = TypeVar("M", bound=BaseModel)


def get_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def publish(client: redis.Redis, stream: str, message: BaseModel, maxlen: int = 100_000) -> str:
    return client.xadd(stream, {"data": message.model_dump_json()}, maxlen=maxlen, approximate=True)


def subscribe(
    client: redis.Redis, stream: str, group: str, consumer: str, model: type[M], block_ms: int = 5000
) -> Iterator[M]:
    """Yield messages for a consumer group, acknowledging each after it is yielded."""
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    while True:
        batches = client.xreadgroup(group, consumer, {stream: ">"}, count=32, block=block_ms)
        for _, entries in batches or []:
            for entry_id, fields in entries:
                yield model.model_validate_json(fields["data"])
                client.xack(stream, group, entry_id)
