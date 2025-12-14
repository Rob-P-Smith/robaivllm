"""
SQLite storage layer for thinking content preservation.

Stores <think> tag content indexed by chat_id with content fingerprints
for verification during re-injection into subsequent conversation turns.

Schema v2: Added content_fingerprint for branch detection.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger("thinking_store")

# Fingerprint length - first N chars of assistant response for verification
FINGERPRINT_LENGTH = 200


class ThinkingStore:
    """SQLite-backed storage for thinking content with fingerprint verification."""

    def __init__(self, db_path: str = None):
        """
        Initialize the thinking store.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "thinking.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema (v2 with fingerprint)."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if we need to migrate (old schema without fingerprint)
            cursor = conn.execute("PRAGMA table_info(thinking_history)")
            columns = [row[1] for row in cursor.fetchall()]

            if "content_fingerprint" not in columns and len(columns) > 0:
                # Old schema exists - drop and recreate
                logger.info("Migrating to v2 schema (dropping old table)")
                conn.execute("DROP TABLE IF EXISTS thinking_history")

            # Create table with fingerprint column
            conn.execute("""
                CREATE TABLE IF NOT EXISTS thinking_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    thinking_content TEXT NOT NULL,
                    content_fingerprint TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chat_id, message_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thinking_chat_id
                ON thinking_history(chat_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thinking_created
                ON thinking_history(created_at)
            """)
            conn.commit()
        logger.info(f"Initialized thinking store v2 at {self.db_path}")

    def store(self, chat_id: str, message_id: str, thinking: str, content: str) -> None:
        """
        Store thinking content with content fingerprint for verification.

        Uses INSERT OR REPLACE to handle duplicate message_ids.

        Args:
            chat_id: Unique chat identifier
            message_id: Unique message identifier
            thinking: The thinking content (without <think> tags)
            content: Full assistant response content (for fingerprint extraction)
        """
        # Extract fingerprint from content (first N chars, excluding thinking tags)
        fingerprint = content[:FINGERPRINT_LENGTH] if content else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO thinking_history
                (chat_id, message_id, thinking_content, content_fingerprint, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (chat_id, message_id, thinking, fingerprint))
            conn.commit()
        logger.debug(f"Stored thinking for chat={chat_id[:8]}..., msg={message_id[:8]}... ({len(thinking)} chars)")

    def store_or_append(self, chat_id: str, message_id: str, thinking: str, content: str) -> bool:
        """
        Store thinking content, appending if entry already exists (for autonomous mode).

        In autonomous mode, multiple LLM turns happen for a single user message.
        Each turn may have thinking content that should be accumulated into one entry.

        Args:
            chat_id: Unique chat identifier
            message_id: Unique message identifier
            thinking: The thinking content (without <think> tags)
            content: Full assistant response content (for fingerprint extraction)

        Returns:
            True if appended to existing, False if created new entry
        """
        fingerprint = content[:FINGERPRINT_LENGTH] if content else ""

        with sqlite3.connect(self.db_path) as conn:
            # Check if entry exists
            cursor = conn.execute("""
                SELECT thinking_content FROM thinking_history
                WHERE chat_id = ? AND message_id = ?
            """, (chat_id, message_id))
            row = cursor.fetchone()

            if row:
                # Append to existing entry with separator
                existing_thinking = row[0]
                combined = f"{existing_thinking}\n\n---\n\n{thinking}"
                conn.execute("""
                    UPDATE thinking_history
                    SET thinking_content = ?, content_fingerprint = ?, created_at = CURRENT_TIMESTAMP
                    WHERE chat_id = ? AND message_id = ?
                """, (combined, fingerprint, chat_id, message_id))
                conn.commit()
                logger.debug(f"Appended thinking for chat={chat_id[:8]}..., msg={message_id[:8]}... (now {len(combined)} chars)")
                return True
            else:
                # Insert new entry
                conn.execute("""
                    INSERT INTO thinking_history
                    (chat_id, message_id, thinking_content, content_fingerprint, created_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (chat_id, message_id, thinking, fingerprint))
                conn.commit()
                logger.debug(f"Created thinking for chat={chat_id[:8]}..., msg={message_id[:8]}... ({len(thinking)} chars)")
                return False

    def get_for_message(self, chat_id: str, message_id: str) -> Optional[str]:
        """
        Get thinking content for a specific message.

        Args:
            chat_id: Unique chat identifier
            message_id: Unique message identifier

        Returns:
            The thinking content, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT thinking_content FROM thinking_history
                WHERE chat_id = ? AND message_id = ?
            """, (chat_id, message_id))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_for_chat(self, chat_id: str) -> dict[str, str]:
        """
        Get all thinking content for a chat.

        Args:
            chat_id: Unique chat identifier

        Returns:
            Dict mapping message_id to thinking content
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT message_id, thinking_content FROM thinking_history
                WHERE chat_id = ?
            """, (chat_id,))
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_ordered_with_fingerprints(self, chat_id: str) -> list[dict]:
        """
        Get thinking entries with fingerprints, ordered by creation time.

        Used for position-based injection with fingerprint verification.

        Args:
            chat_id: Unique chat identifier

        Returns:
            List of dicts with 'thinking', 'fingerprint', 'id' keys
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT thinking_content, content_fingerprint, id
                FROM thinking_history
                WHERE chat_id = ?
                ORDER BY created_at ASC
            """, (chat_id,))
            return [
                {"thinking": row[0], "fingerprint": row[1], "id": row[2]}
                for row in cursor.fetchall()
            ]

    def prune_from_position(self, chat_id: str, position: int) -> int:
        """
        Delete orphaned entries from position onwards.

        Called when conversation branching is detected - removes thinking
        entries that no longer correspond to messages in the conversation.

        Args:
            chat_id: Unique chat identifier
            position: Position index (0-based) from which to start pruning

        Returns:
            Number of records deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get IDs of entries at position and later
            cursor = conn.execute("""
                SELECT id FROM thinking_history
                WHERE chat_id = ?
                ORDER BY created_at ASC
                LIMIT -1 OFFSET ?
            """, (chat_id, position))
            ids_to_delete = [row[0] for row in cursor.fetchall()]

            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(f"""
                    DELETE FROM thinking_history
                    WHERE id IN ({placeholders})
                """, ids_to_delete)
                conn.commit()

            return len(ids_to_delete)

    def delete_chat(self, chat_id: str) -> int:
        """
        Delete all thinking content for a chat.

        Args:
            chat_id: Unique chat identifier

        Returns:
            Number of records deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM thinking_history WHERE chat_id = ?
            """, (chat_id,))
            conn.commit()
            deleted = cursor.rowcount
        logger.info(f"Deleted {deleted} records for chat {chat_id}")
        return deleted

    def cleanup_old(self, days: int = 90) -> int:
        """
        Delete thinking content older than specified days.

        Args:
            days: Number of days to retain (default 90)

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM thinking_history
                WHERE created_at < ?
            """, (cutoff.isoformat(),))
            conn.commit()
            deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} records older than {days} days")
        return deleted

    def get_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dict with chat_count, message_count, oldest_record, newest_record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(DISTINCT chat_id) as chat_count,
                    COUNT(*) as message_count,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM thinking_history
            """)
            row = cursor.fetchone()
            return {
                "chat_count": row[0],
                "message_count": row[1],
                "oldest_record": row[2],
                "newest_record": row[3]
            }
