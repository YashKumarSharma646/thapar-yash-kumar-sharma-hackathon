"""Thin wrapper over Redis Streams: publish models, consume batches with consumer groups."""

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


def publish_many(client: redis.Redis, stream: str, messages: list[BaseModel], maxlen: int = 100_000) -> None:
    pipe = client.pipeline(transaction=False)
    for m in messages:
        pipe.xadd(stream, {"data": m.model_dump_json()}, maxlen=maxlen, approximate=True)
    pipe.execute()


def consume_batches(
    client: redis.Redis,
    stream: str,
    group: str,
    consumer: str,
    model: type[M],
    batch_size: int = 64,
    block_ms: int = 2000,
    start: str = "0",
) -> Iterator[list[M]]:
    """Yield batches for a consumer group. A batch is acknowledged once the caller asks for the next one,
    so a crash mid-batch leaves it pending for redelivery. `start` applies when the group is first created
    ("0": the whole stream, "$": only new messages); a group lost to a stream reset restarts from "0"."""
    def ensure_group(start_id: str = "0") -> None:
        try:
            client.xgroup_create(stream, group, id=start_id, mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    ensure_group(start)
    while True:
        try:
            response = client.xreadgroup(group, consumer, {stream: ">"}, count=batch_size, block=block_ms)
        except redis.ResponseError as e:
            if "NOGROUP" not in str(e):
                raise
            ensure_group()  # the stream was deleted (a new replay session), so the group must be recreated
            continue
        entries = [entry for _, stream_entries in response or [] for entry in stream_entries]
        if not entries:
            continue
        yield [model.model_validate_json(fields["data"]) for _, fields in entries]
        client.xack(stream, group, *[entry_id for entry_id, _ in entries])
