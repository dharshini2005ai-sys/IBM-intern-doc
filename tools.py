from typing import Dict, Any

def calculate_attendance(attended: int, total: int) -> Dict[str, Any]:
    if total <= 0:
        raise ValueError("Total classes must be greater than zero.")
    if attended < 0:
        raise ValueError("Attended classes cannot be negative.")
    if attended > total:
        raise ValueError("Attended classes cannot exceed total classes.")
    return {
        "attended": attended,
        "total": total,
        "percentage": round((attended / total) * 100, 2),
    }

def calculate_expression(expression: str) -> float:
    allowed = set("0123456789+-*/(). %")
    if not expression:
        raise ValueError("Expression is empty.")
    if any(char not in allowed for char in expression):
        raise ValueError("Expression contains unsupported characters.")
    expression = expression.replace("%", "/100")
    try:
        return float(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        raise ValueError("Invalid mathematical expression.") from exc
