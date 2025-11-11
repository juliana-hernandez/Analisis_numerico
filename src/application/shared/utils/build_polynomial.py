## Recibe una lista de coeficientes de la forma [a0, a1, a2, ..., an] y devuelve un string con el polinomio correspondiente en su forma canónica.
def build_polynomial(coefficients):
    n = len(coefficients)
    polynomial_terms = []
    polynomial = ""
    for i in range(n):
        if i == 0:
            if coefficients[i] < 0:
                polynomial_terms.append(f"-{abs(coefficients[i])}")
            else:
                polynomial_terms.append(f"+{abs(coefficients[i])}")
        else:
            polynomial_terms.append(f"+{coefficients[i]}*x**{i} ")

    polynomial_terms.reverse()
    polynomial = "".join(polynomial_terms)
    polynomial = polynomial.replace("+-", "-")
    return polynomial
