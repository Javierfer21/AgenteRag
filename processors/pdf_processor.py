"""
PDF document processor using PyPDF2.
"""
import io

from processors.base import BaseProcessor


class PDFProcessor(BaseProcessor):
    """Extracts plain text from PDF files using PyPDF2."""

    def process(self, file_bytes: bytes) -> str:
        """Extract text from a PDF file.

        Args:
            file_bytes: Raw bytes of the PDF document.

        Returns:
            Concatenated text from all pages.

        Raises:
            ImportError: If PyPDF2 is not installed.
            Exception: If the PDF cannot be read.
        """
        try:
            import PyPDF2
        except ImportError as exc:
            raise ImportError(
                "PyPDF2 is required for PDF processing. "
                "Install it with: pip install PyPDF2"
            ) from exc

        text_parts: list[str] = []
        pdf_file = io.BytesIO(file_bytes)

        reader = PyPDF2.PdfReader(pdf_file)
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                # Skip pages that fail extraction but continue with the rest
                text_parts.append(f"[Página {page_num + 1}: error de extracción — {e}]")

        return "\n\n".join(text_parts)
