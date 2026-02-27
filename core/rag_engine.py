"""
Neuro-Lite RAG Engine
─────────────────────
SQLite FTS5 full-text search.
WAL mode for concurrent reads.
Sub-10ms retrieval target.
"""

import sqlite3
import logging
import threading
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("neurolite.rag")

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class SearchResult:
    rowid: int
    topic: str
    question: str
    answer: str
    source: str
    rank: float
    created_at: str


class RAGEngine:
    """Thread-safe SQLite FTS5 retrieval engine."""

    _instance: Optional["RAGEngine"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if self._initialized:
            return

        if db_path is None:
            db_path = os.environ.get(
                "KNOWLEDGE_DB",
                str(BASE_DIR / "data" / "knowledge.db"),
            )

        self._db_path = str(Path(db_path).resolve())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._write_lock = threading.Lock()

        # Initialize schema on first connection
        conn = self._get_connection()
        self._create_schema(conn)
        self._initialized = True
        logger.info("RAG engine initialized: %s", self._db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self._db_path,
                timeout=10.0,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-4000")  # 4MB cache
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create tables if they don't exist."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                topic,
                question,
                answer,
                content='knowledge',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(rowid, topic, question, answer)
                VALUES (new.id, new.topic, new.question, new.answer);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, topic, question, answer)
                VALUES ('delete', old.id, old.topic, old.question, old.answer);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, topic, question, answer)
                VALUES ('delete', old.id, old.topic, old.question, old.answer);
                INSERT INTO knowledge_fts(rowid, topic, question, answer)
                VALUES (new.id, new.topic, new.question, new.answer);
            END;
        """)
        conn.commit()

    def search(self, query: str, limit: int = 3) -> List[SearchResult]:
        """
        FTS5 search with BM25 ranking.
        Target: sub-10ms.
        """
        if not query or not query.strip():
            return []

        start = time.monotonic()
        conn = self._get_connection()

        # Clean query for FTS5 syntax
        safe_query = self._sanitize_query(query)
        if not safe_query:
            return []

        try:
            cursor = conn.execute(
                """
                SELECT k.id, k.topic, k.question, k.answer, k.source,
                       k.created_at, rank
                FROM knowledge_fts fts
                JOIN knowledge k ON k.id = fts.rowid
                WHERE knowledge_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query failed: %s (query='%s')", e, safe_query)
            return []

        results = [
            SearchResult(
                rowid=row["id"],
                topic=row["topic"],
                question=row["question"],
                answer=row["answer"],
                source=row["source"],
                rank=row["rank"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug("RAG search '%.40s' → %d results in %.1fms", query, len(results), elapsed_ms)
        return results

    def _sanitize_query(self, query: str) -> str:
        """Sanitize query for FTS5 MATCH syntax."""
        # Remove FTS5 operators and special chars
        cleaned = query.strip()
        # Extract meaningful words
        import re
        words = re.findall(r"[a-zA-Z0-9\u00C0-\u024F\u4e00-\u9fff]+", cleaned)
        if not words:
            return ""
        # Use OR to be more permissive
        return " OR ".join(words[:10])  # Limit to 10 terms

    def add_entry(
        self,
        question: str,
        answer: str,
        topic: str = "",
        source: str = "manual",
    ) -> int:
        """Add a knowledge entry. Returns the new row ID."""
        with self._write_lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "INSERT INTO knowledge (topic, question, answer, source) VALUES (?, ?, ?, ?)",
                (topic.strip(), question.strip(), answer.strip(), source),
            )
            conn.commit()
            logger.info("Knowledge added: id=%d topic='%s'", cursor.lastrowid, topic)
            return cursor.lastrowid

    def add_entries_batch(self, entries: List[Dict[str, str]]) -> int:
        """Batch insert. Returns count of inserted rows."""
        with self._write_lock:
            conn = self._get_connection()
            inserted = 0
            for entry in entries:
                try:
                    conn.execute(
                        "INSERT INTO knowledge (topic, question, answer, source) VALUES (?, ?, ?, ?)",
                        (
                            entry.get("topic", "").strip(),
                            entry["question"].strip(),
                            entry["answer"].strip(),
                            entry.get("source", "batch"),
                        ),
                    )
                    inserted += 1
                except (KeyError, sqlite3.Error) as e:
                    logger.warning("Batch insert skip: %s", e)
            conn.commit()
            logger.info("Batch inserted: %d/%d entries", inserted, len(entries))
            return inserted

    def delete_entry(self, entry_id: int) -> bool:
        """Delete by ID."""
        with self._write_lock:
            conn = self._get_connection()
            cursor = conn.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_all(self, limit: int = 500, offset: int = 0) -> List[Dict[str, Any]]:
        """List all entries with pagination."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT id, topic, question, answer, source, created_at FROM knowledge ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count(self) -> int:
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM knowledge")
        return cursor.fetchone()[0]

    def export_all(self) -> List[Dict[str, str]]:
        """Export all knowledge for sharing."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT topic, question, answer, source FROM knowledge ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]

    def rebuild_fts(self) -> None:
        """Rebuild FTS index."""
        with self._write_lock:
            conn = self._get_connection()
            conn.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES ('rebuild')")
            conn.commit()
            logger.info("FTS index rebuilt.")
