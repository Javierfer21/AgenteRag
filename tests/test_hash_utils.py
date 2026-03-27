"""
Tests for hashing utility functions.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils.hash_utils import md5_hash, file_hash


class TestMd5Hash:
    def test_returns_string(self):
        result = md5_hash("hello")
        assert isinstance(result, str)

    def test_returns_32_char_hex(self):
        result = md5_hash("hello")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_input_same_hash(self):
        assert md5_hash("test_input") == md5_hash("test_input")

    def test_different_inputs_different_hashes(self):
        assert md5_hash("input_a") != md5_hash("input_b")

    def test_empty_string(self):
        result = md5_hash("")
        assert isinstance(result, str)
        assert len(result) == 32

    def test_known_hash(self):
        # MD5 of "hello" is well-known
        assert md5_hash("hello") == "5d41402abc4b2a76b9719d911017c592"

    def test_unicode_input(self):
        result = md5_hash("hola mundo 🌍")
        assert isinstance(result, str)
        assert len(result) == 32

    def test_long_string(self):
        long_text = "A" * 100_000
        result = md5_hash(long_text)
        assert isinstance(result, str)
        assert len(result) == 32


class TestFileHash:
    def test_returns_string(self):
        result = file_hash(b"some bytes")
        assert isinstance(result, str)

    def test_returns_32_char_hex(self):
        result = file_hash(b"some bytes")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_bytes_same_hash(self):
        data = b"\x00\x01\x02\x03"
        assert file_hash(data) == file_hash(data)

    def test_different_bytes_different_hashes(self):
        assert file_hash(b"data_one") != file_hash(b"data_two")

    def test_empty_bytes(self):
        result = file_hash(b"")
        assert isinstance(result, str)
        assert len(result) == 32

    def test_binary_data(self):
        data = bytes(range(256))
        result = file_hash(data)
        assert isinstance(result, str)
        assert len(result) == 32
