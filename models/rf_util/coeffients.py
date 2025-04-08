import numpy as np
from sympy import *

def maclaurin_coeffiencts(f, x, n=20):
    coefs = [f.evalf(subs={x:0})]
    df_i = diff(f, x)
    for i in range(1, n+1):
        coefs.append(df_i.evalf(subs={x:0})/factorial(i))
        df_i = diff(df_i, x)
    return coefs

def hyperdiff_functions(f, x, n=20):
    functions = [f]
    df_i = diff(f, x)
    for _ in range(1, n+1):
        functions.append(df_i)
        df_i = diff(df_i, x)
    return functions

class Maclaurin_Coefs():
    """
    return hyper diff value
    """
    def __init__(self, style = 'exp', n = 20):
        self.x = symbols('x')
        self.functions = {
            'exp': exp(self.x),
            'inverse': 1/(1-self.x),
            'log': 1-log(1-self.x),
            'trigh': cosh(self.x) + sinh(self.x),
            'sqrt': 2-sqrt(1-self.x)
        }
        self.f = self.functions[style]
        self.diffs = hyperdiff_functions(self.f, self.x, n=n)
        self.diff_values = maclaurin_coeffiencts(self.f, self.x, n=n)

    def coefs(self, k):
        if type(k) == int:
            return self.diff_values[k]
        return np.array([self.diff_values[v] for v in k],dtype=np.float32)

