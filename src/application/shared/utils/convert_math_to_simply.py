# Función auxiliar para convertir expresiones de `math` a `SymPy`
def convert_math_to_sympy(expr):
    replacements = {
        "math.exp": "exp",
        "math.log": "log",
        "math.sqrt": "sqrt",
        "math.sin": "sin",
        "math.cos": "cos",
        "math.tan": "tan",
        "math.asin": "asin",
        "math.acos": "acos",
        "math.atan": "atan",
        "math.pi": "pi",
        "math.e": "E",
    }
    for math_func, sympy_func in replacements.items():
        expr = expr.replace(math_func, sympy_func)
    return expr
