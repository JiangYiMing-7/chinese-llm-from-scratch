#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for large-scale data preprocessing."""

import logging
import os

logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path when it is not present."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_distinct_paths(*paths: str) -> None:
    """Refuse accidental in-place writes that would destroy the input file."""
    resolved = [os.path.abspath(p) for p in paths if p]
    if len(resolved) != len(set(resolved)):
        raise ValueError("input/output/report paths must be different files")


def reset_redis_prefix(redis_host: str, redis_port: int, redis_prefix: str) -> int:
    """Delete Redis keys that belong to one preprocessing run prefix."""
    import redis

    client = redis.Redis(host=redis_host, port=redis_port)
    client.ping()
    pattern = f"{redis_prefix}*"
    deleted = 0
    batch = []
    for key in client.scan_iter(match=pattern, count=1000):
        batch.append(key)
        if len(batch) >= 1000:
            deleted += client.delete(*batch)
            batch.clear()
    if batch:
        deleted += client.delete(*batch)
    logger.info("已清理 Redis 前缀 %r 下的 %d 个键", redis_prefix, deleted)
    return deleted


class ExactHashStore:
    """Exact SHA-256 dedup storage.

    Memory mode keeps a local Python set for small runs. Redis mode stores hashes
    outside the Python process, matching the MinHashLSH Redis backend for large
    corpora.
    """

    def __init__(
        self,
        backend: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_prefix: str = "dedup",
    ):
        self.backend = backend
        self._set = set()
        self._redis = None
        self._redis_key = None

        if backend == "redis":
            import redis

            self._redis = redis.Redis(host=redis_host, port=redis_port)
            self._redis.ping()
            self._redis_key = f"{redis_prefix}:exact_hashes"
            logger.info("SHA-256 精确去重后端：redis key=%s", self._redis_key)
        else:
            logger.info("SHA-256 精确去重后端：memory")

    def contains(self, digest: str) -> bool:
        if self.backend == "redis":
            return bool(self._redis.sismember(self._redis_key, digest))
        return digest in self._set

    def add(self, digest: str) -> None:
        if self.backend == "redis":
            self._redis.sadd(self._redis_key, digest)
        else:
            self._set.add(digest)

    def __len__(self) -> int:
        if self.backend == "redis":
            return int(self._redis.scard(self._redis_key))
        return len(self._set)
