"""
Tests for the SQLite-based ConversationMemory class.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.memory import ConversationMemory


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def memory(temp_db):
    """Return a ConversationMemory instance with a temporary DB."""
    return ConversationMemory(db_path=temp_db, user_id="test-user-001")


class TestConversationMemoryInit:
    def test_creates_db_file(self, temp_db):
        ConversationMemory(db_path=temp_db, user_id="user1")
        assert os.path.exists(temp_db)

    def test_creates_directory_if_missing(self, tmp_path):
        nested_path = str(tmp_path / "nested" / "dir" / "memory.db")
        ConversationMemory(db_path=nested_path, user_id="user1")
        assert os.path.exists(nested_path)


class TestSaveAndRetrieve:
    def test_save_single_message(self, memory):
        memory.save_message("user", "Hello!")
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"

    def test_save_multiple_messages(self, memory):
        memory.save_message("user", "Question 1")
        memory.save_message("assistant", "Answer 1")
        memory.save_message("user", "Question 2")
        history = memory.get_history()
        assert len(history) == 3

    def test_messages_in_chronological_order(self, memory):
        memory.save_message("user", "First")
        memory.save_message("assistant", "Second")
        memory.save_message("user", "Third")
        history = memory.get_history()
        assert history[0]["content"] == "First"
        assert history[1]["content"] == "Second"
        assert history[2]["content"] == "Third"

    def test_get_history_with_limit(self, memory):
        for i in range(10):
            memory.save_message("user", f"Message {i}")
        history = memory.get_history(limit=5)
        assert len(history) == 5

    def test_history_contains_required_keys(self, memory):
        memory.save_message("user", "Test")
        msg = memory.get_history()[0]
        assert "role" in msg
        assert "content" in msg
        assert "timestamp" in msg

    def test_empty_history(self, memory):
        assert memory.get_history() == []

    def test_user_isolation(self, temp_db):
        """Messages from different users should not mix."""
        mem_a = ConversationMemory(db_path=temp_db, user_id="user-A")
        mem_b = ConversationMemory(db_path=temp_db, user_id="user-B")

        mem_a.save_message("user", "Message for A")
        mem_b.save_message("user", "Message for B")

        assert len(mem_a.get_history()) == 1
        assert len(mem_b.get_history()) == 1
        assert mem_a.get_history()[0]["content"] == "Message for A"
        assert mem_b.get_history()[0]["content"] == "Message for B"


class TestClearHistory:
    def test_clear_removes_all_messages(self, memory):
        memory.save_message("user", "To be cleared")
        memory.save_message("assistant", "Also cleared")
        memory.clear_history()
        assert memory.get_history() == []

    def test_clear_only_affects_own_user(self, temp_db):
        mem_a = ConversationMemory(db_path=temp_db, user_id="user-A")
        mem_b = ConversationMemory(db_path=temp_db, user_id="user-B")

        mem_a.save_message("user", "A's message")
        mem_b.save_message("user", "B's message")

        mem_a.clear_history()

        assert mem_a.get_history() == []
        assert len(mem_b.get_history()) == 1

    def test_clear_then_save(self, memory):
        memory.save_message("user", "Old message")
        memory.clear_history()
        memory.save_message("user", "New message")
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["content"] == "New message"
