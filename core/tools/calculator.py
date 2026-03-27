"""
Safe calculator tool using AST-based expression evaluation.
No use of eval() — only whitelisted AST node types are processed.
"""
from __future__ import annotations

import ast
import operator
from typing import Union

from langchain_core.tools import tool

# Supported binary operators
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# Supported unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

Number = Union[int, float]


def _safe_eval(node: ast.AST) -> Number:
    """Recursively evaluate an AST node using only safe operations.

    Args:
        node: An AST node to evaluate.

    Returns:
        Numeric result of the expression.

    Raises:
        ValueError: If the node type is not supported.
        ZeroDivisionError: On division by zero.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Tipo de constante no soportado: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Operador binario no soportado: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("División por cero no está permitida.")
        return _BINARY_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Operador unario no soportado: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _UNARY_OPS[op_type](operand)

    raise ValueError(f"Tipo de nodo no soportado: {type(node).__name__}")


@tool
def calculadora(expression: str) -> str:
    """Evalúa una expresión matemática de forma segura sin usar eval().

    Soporta las operaciones: suma (+), resta (-), multiplicación (*),
    división (/), potencia (**), módulo (%), división entera (//).

    Args:
        expression: Expresión matemática como cadena de texto.
                    Ejemplos: '2 + 3', '10 / 4', '2 ** 8', '17 % 5'

    Returns:
        Resultado numérico como cadena de texto, o mensaje de error.
    """
    expression = expression.strip()
    if not expression:
        return "Error: expresión vacía."

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)

        # Return integer representation when possible
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    except ZeroDivisionError as e:
        return f"Error: {e}"
    except ValueError as e:
        return f"Error: expresión no válida — {e}"
    except SyntaxError:
        return f"Error: sintaxis inválida en '{expression}'."
    except Exception as e:
        return f"Error inesperado al calcular '{expression}': {e}"
