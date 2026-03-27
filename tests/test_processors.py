"""
Tests for document processors and the processor factory.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from processors import get_processor
from processors.txt_processor import TxtProcessor
from processors.csv_processor import CSVProcessor


class TestTxtProcessor:
    def test_process_utf8_bytes(self):
        text = "Hello, this is a UTF-8 encoded text."
        result = TxtProcessor().process(text.encode("utf-8"))
        assert result == text

    def test_process_latin1_bytes(self):
        text = "Caféé résumé naïve"
        result = TxtProcessor().process(text.encode("latin-1"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_process_empty_bytes(self):
        result = TxtProcessor().process(b"")
        assert result == ""

    def test_process_multiline(self):
        text = "Line 1\nLine 2\nLine 3"
        result = TxtProcessor().process(text.encode("utf-8"))
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_process_returns_string(self):
        result = TxtProcessor().process(b"Any content")
        assert isinstance(result, str)


class TestCSVProcessor:
    def _make_csv_bytes(self, content: str) -> bytes:
        return content.encode("utf-8")

    def test_basic_csv_processing(self):
        csv_content = "name,age,city\nAlice,30,Madrid\nBob,25,Barcelona\n"
        result = CSVProcessor().process(self._make_csv_bytes(csv_content))
        assert isinstance(result, str)
        assert "name" in result
        assert "Alice" in result

    def test_returns_shape_info(self):
        csv_content = "col1,col2\n1,2\n3,4\n5,6\n"
        result = CSVProcessor().process(self._make_csv_bytes(csv_content))
        assert "3" in result  # 3 rows
        assert "2" in result  # 2 columns

    def test_numeric_statistics_included(self):
        csv_content = "value\n10\n20\n30\n40\n50\n"
        result = CSVProcessor().process(self._make_csv_bytes(csv_content))
        # describe() output should include count/mean/std
        assert "count" in result.lower() or "mean" in result.lower() or "30" in result

    def test_empty_csv(self):
        csv_content = "col1,col2\n"
        result = CSVProcessor().process(self._make_csv_bytes(csv_content))
        assert isinstance(result, str)

    def test_csv_with_spaces_in_values(self):
        csv_content = "name,description\nItem A,A description here\n"
        result = CSVProcessor().process(self._make_csv_bytes(csv_content))
        assert "Item A" in result


class TestGetProcessor:
    def test_pdf_extension(self):
        processor = get_processor("pdf")
        from processors.pdf_processor import PDFProcessor
        assert isinstance(processor, PDFProcessor)

    def test_docx_extension(self):
        processor = get_processor("docx")
        from processors.docx_processor import DocxProcessor
        assert isinstance(processor, DocxProcessor)

    def test_txt_extension(self):
        processor = get_processor("txt")
        assert isinstance(processor, TxtProcessor)

    def test_md_extension(self):
        processor = get_processor("md")
        assert isinstance(processor, TxtProcessor)

    def test_log_extension(self):
        processor = get_processor("log")
        assert isinstance(processor, TxtProcessor)

    def test_csv_extension(self):
        processor = get_processor("csv")
        assert isinstance(processor, CSVProcessor)

    def test_xlsx_extension(self):
        processor = get_processor("xlsx")
        from processors.excel_processor import ExcelProcessor
        assert isinstance(processor, ExcelProcessor)

    def test_xls_extension(self):
        processor = get_processor("xls")
        from processors.excel_processor import ExcelProcessor
        assert isinstance(processor, ExcelProcessor)

    def test_unsupported_extension_returns_none(self):
        assert get_processor("exe") is None
        assert get_processor("zip") is None
        assert get_processor("jpg") is None

    def test_case_insensitive_extension(self):
        assert get_processor("PDF") is not None
        assert get_processor("DOCX") is not None
        assert get_processor("CSV") is not None

    def test_extension_with_leading_dot(self):
        # Should handle '.pdf' as well as 'pdf'
        assert get_processor(".pdf") is not None
