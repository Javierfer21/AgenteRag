"""
SQLite-based conversation memory for persisting chat history.
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional


class ConversationMemory:
    """Persists and retrieves conversation history using SQLite."""

    def __init__(self, db_path: str, user_id: str) -> None:
        """Initialize the database connection and create tables if needed.

        Args:
            db_path: Path to the SQLite database file.
            user_id: Unique identifier for the current user/session.
        """
        self.db_path = db_path
        self.user_id = user_id
        self._ensure_directory()
        self._create_tables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_directory(self) -> None:
        """Create parent directories for the database file if they don't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection with row_factory set."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        """Create the messages table if it doesn't already exist."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT    NOT NULL,
            role      TEXT    NOT NULL,
            content   TEXT    NOT NULL,
            timestamp TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_user_id
            ON messages (user_id);
        """
        with self._get_connection() as conn:
            conn.executescript(create_sql)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_message(self, role: str, content: str) -> None:
        """Persist a single message to the database.

        Args:
            role: Message role, e.g. 'user' or 'assistant'.
            content: Text content of the message.
        """
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (self.user_id, role, content, timestamp),
            )

    def get_history(self, limit: int = 20) -> list[dict]:
        """Retrieve the most recent messages for the current user.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            List of message dicts with keys: id, role, content, timestamp.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp
                FROM messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.user_id, limit),
            ).fetchall()

        # Return in chronological order
        return [dict(row) for row in reversed(rows)]

    def clear_history(self) -> None:
        """Delete all messages for the current user."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM messages WHERE user_id = ?",
                (self.user_id,),
            )
