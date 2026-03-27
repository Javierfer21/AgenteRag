"""
Tests for the safe AST-based calculator tool.
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.tools.calculator import calculadora


def _call(expr: str) -> str:
    """Helper: invoke the tool function directly by its underlying callable."""
    return calculadora.invoke({"expression": expr})


class TestBasicOperations:
    def test_addition(self):
        assert _call("2 + 3") == "5"

    def test_subtraction(self):
        assert _call("10 - 4") == "6"

    def test_multiplication(self):
        assert _call("6 * 7") == "42"

    def test_division_exact(self):
        assert _call("10 / 2") == "5"

    def test_division_float(self):
        result = _call("7 / 2")
        assert result == "3.5"

    def test_power(self):
        assert _call("2 ** 10") == "1024"

    def test_modulo(self):
        assert _call("17 % 5") == "2"

    def test_floor_division(self):
        assert _call("17 // 5") == "3"

    def test_unary_minus(self):
        assert _call("-5 + 10") == "5"

    def test_parentheses_precedence(self):
        assert _call("(2 + 3) * 4") == "20"

    def test_nested_expression(self):
        assert _call("2 ** 3 + 4 * 5 - 1") == "27"


class TestDivisionByZero:
    def test_division_by_zero_returns_error(self):
        result = _call("10 / 0")
        assert "Error" in result or "error" in result.lower()
        assert "cero" in result.lower() or "zero" in result.lower() or "0" in result

    def test_modulo_by_zero_returns_error(self):
        result = _call("10 % 0")
        assert "Error" in result or "error" in result.lower()

    def test_floor_division_by_zero_returns_error(self):
        result = _call("10 // 0")
        assert "Error" in result or "error" in result.lower()


class TestInvalidExpressions:
    def test_empty_expression(self):
        result = _call("")
        assert "Error" in result or "error" in result.lower()

    def test_string_expression(self):
        result = _call("hello + world")
        assert "Error" in result or "error" in result.lower()

    def test_import_attempt(self):
        result = _call("__import__('os')")
        assert "Error" in result or "error" in result.lower()

    def test_exec_attempt(self):
        result = _call("exec('print(1)')")
        assert "Error" in result or "error" in result.lower()

    def test_invalid_syntax(self):
        result = _call("2 +* 3")
        assert "Error" in result or "error" in result.lower()

    def test_whitespace_only(self):
        result = _call("   ")
        assert "Error" in result or "error" in result.lower()
