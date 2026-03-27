"""
Processors package.
Provides a factory function to select the right document processor by extension.
"""
from __future__ import annotations

from typing import Optional

from processors.base import BaseProcessor


def get_processor(file_extension: str) -> Optional[BaseProcessor]:
    """Return the appropriate processor instance for a given file extension.

    Args:
        file_extension: File extension without leading dot (e.g. 'pdf', 'docx').

    Returns:
        A BaseProcessor subclass instance, or None if unsupported.
    """
    ext = file_extension.lower().strip(".")

    if ext == "pdf":
        from processors.pdf_processor import PDFProcessor
        return PDFProcessor()

    if ext == "docx":
        from processors.docx_processor import DocxProcessor
        return DocxProcessor()

    if ext in ("txt", "md", "log"):
        from processors.txt_processor import TxtProcessor
        return TxtProcessor()

    if ext == "csv":
        from processors.csv_processor import CSVProcessor
        return CSVProcessor()

    if ext in ("xlsx", "xls"):
        from processors.excel_processor import ExcelProcessor
        return ExcelProcessor()

    return None
