"""
Tests for the text chunking utilities.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils.chunking import chunk_text, chunk_document


class TestChunkText:
    def test_basic_chunking(self):
        text = "A" * 2500
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 1000

    def test_short_text_single_chunk(self):
        text = "Hello, world!"
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_present(self):
        """Verify that consecutive chunks share overlapping content."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 20  # 520 chars
        chunk_size = 100
        overlap = 40
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=overlap)
        if len(chunks) >= 2:
            # The end of chunk[0] should appear at the start of chunk[1]
            tail = chunks[0][-overlap:]
            assert chunks[1].startswith(tail)

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=100, chunk_overlap=-1)

    def test_overlap_ge_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=100, chunk_overlap=100)

    def test_exact_size_text(self):
        text = "X" * 1000
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestChunkDocument:
    def test_returns_list_of_dicts(self):
        text = "Word " * 500  # 2500 chars
        result = chunk_document(text, filename="test.txt", chunk_size=500, chunk_overlap=100)
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], dict)

    def test_required_keys_present(self):
        text = "Hello world! " * 100
        chunks = chunk_document(text, filename="doc.pdf", chunk_size=200, chunk_overlap=50)
        for chunk in chunks:
            assert "text" in chunk
            assert "filename" in chunk
            assert "chunk_index" in chunk
            assert "chunk_id" in chunk

    def test_filename_matches(self):
        text = "Content " * 200
        filename = "myfile.docx"
        chunks = chunk_document(text, filename=filename, chunk_size=300, chunk_overlap=50)
        for chunk in chunks:
            assert chunk["filename"] == filename

    def test_chunk_index_is_sequential(self):
        text = "Data " * 500
        chunks = chunk_document(text, filename="data.txt", chunk_size=400, chunk_overlap=100)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_chunk_id_is_string(self):
        text = "Sample text for hashing."
        chunks = chunk_document(text, filename="sample.txt", chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert isinstance(chunk["chunk_id"], str)
            assert len(chunk["chunk_id"]) == 32  # MD5 hex digest

    def test_empty_text_returns_empty_list(self):
        result = chunk_document("", filename="empty.txt")
        assert result == []

    def test_chunk_ids_are_unique(self):
        text = "Unique content " * 300
        chunks = chunk_document(text, filename="unique.txt", chunk_size=300, chunk_overlap=60)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"
