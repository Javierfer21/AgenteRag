"""
Plain-text document processor (TXT, MD, LOG files).
"""
from processors.base import BaseProcessor


class TxtProcessor(BaseProcessor):
    """Decodes plain text files from bytes."""

    def process(self, file_bytes: bytes) -> str:
        """Decode text from raw bytes, attempting UTF-8 then Latin-1.

        Args:
            file_bytes: Raw bytes of the text file.

        Returns:
            Decoded string content.
        """
        # Try UTF-8 first (most common), fall back to Latin-1 which never fails
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")
