"""Saves finished scans so they can be shown later on /report/<id>.

Uses Postgres when DATABASE_URL is set (production on Railway). Without it,
falls back to an in-memory dict so the app still works for local testing -
but in-memory reports disappear on every restart or redeploy.
"""
import json
import secrets
from typing import Optional

import asyncpg

import config

_pool = None
_memory = {}


async def init() -> None:
    global _pool
    if not config.DATABASE_URL:
        print("[storage] DATABASE_URL not set - reports will be kept in memory and lost on restart")
        return
    _pool = await asyncpg.create_pool(config.DATABASE_URL, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                business_name TEXT,
                data JSONB NOT NULL
            )
            """
        )
    print("[storage] connected to Postgres")


async def close() -> None:
    if _pool is not None:
        await _pool.close()


def new_report_id() -> str:
    # ~128 bits of randomness, so report links can't be guessed.
    return secrets.token_urlsafe(16)


async def save_report(report_id: str, business_name: str, data: dict) -> None:
    if _pool is None:
        _memory[report_id] = data
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO reports (id, business_name, data) VALUES ($1, $2, $3::jsonb)",
            report_id,
            business_name,
            json.dumps(data),
        )


async def get_report(report_id: str) -> Optional[dict]:
    if _pool is None:
        return _memory.get(report_id)
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("SELECT data FROM reports WHERE id = $1", report_id)
    if row is None:
        return None
    data = row["data"]
    return json.loads(data) if isinstance(data, str) else data
