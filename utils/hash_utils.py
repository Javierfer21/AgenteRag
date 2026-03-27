"""
Hashing utilities used across the application.
"""
import hashlib


def md5_hash(text: str) -> str:
    """Return the MD5 hex digest of a string.

    Args:
        text: Input string to hash.

    Returns:
        32-character lowercase hexadecimal MD5 digest.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def file_hash(file_bytes: bytes) -> str:
    """Return the MD5 hex digest of raw file bytes.

    Args:
        file_bytes: Raw binary content of a file.

    Returns:
        32-character lowercase hexadecimal MD5 digest.
    """
    return hashlib.md5(file_bytes).hexdigest()
