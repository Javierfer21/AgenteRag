"""
Abstract base class for all document processors.
"""
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Base interface for document text extractors."""

    @abstractmethod
    def process(self, file_bytes: bytes) -> str:
        """Extract plain text from raw file bytes.

        Args:
            file_bytes: Raw binary content of the document.

        Returns:
            Extracted plain text as a single string.

        Raises:
            Exception: If text extraction fails.
        """
