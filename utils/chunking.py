"""
Text chunking utilities for splitting documents into overlapping segments.
"""
from __future__ import annotations

from utils.hash_utils import md5_hash


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split a text string into overlapping chunks.

    The chunking operates on characters (not tokens). Each chunk has
    approximately `chunk_size` characters, with `chunk_overlap` characters
    shared between consecutive chunks.

    Args:
        text: The input text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunk strings. Empty if input text is empty or whitespace.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= text_length:
            break
        start = end - chunk_overlap

    return chunks


def chunk_document(
    text: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Split a document into overlapping chunks with metadata.

    Args:
        text: Full document text to split.
        filename: Original filename of the document.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of dicts, each containing:
            - text (str): Chunk text content.
            - filename (str): Source document filename.
            - chunk_index (int): Zero-based position of the chunk.
            - chunk_id (str): MD5 hash of filename + chunk_index + text.
    """
    raw_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    result: list[dict] = []
    for idx, chunk_text_str in enumerate(raw_chunks):
        chunk_id = md5_hash(f"{filename}_{idx}_{chunk_text_str}")
        result.append({
            "text": chunk_text_str,
            "filename": filename,
            "chunk_index": idx,
            "chunk_id": chunk_id,
        })

    return result
