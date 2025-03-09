
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scienceplots

from sympy import symbols
import sympy

np.set_printoptions(suppress=True, linewidth=200, precision=3)


def draw_marginal_coeff(past_x0_coeff, past_eps_coeff, node_coeff, path):
    with plt.style.context(['science', 'ieee']):
        fig, ax = plt.subplots()
        ax.plot(node_coeff[1:, 0], past_x0_coeff.sum(axis=1), label="ideal signal", linestyle="solid", color="orange")
        ax.plot(node_coeff[1:, 0], node_coeff[1:, 1],  label="equiv signal", linestyle="dashed", color="red")
        ax.plot(node_coeff[1:, 0], np.linalg.norm(past_eps_coeff, axis=1), label="ideal noise", linestyle="solid", color="magenta")
        ax.plot(node_coeff[1:, 0], node_coeff[1:, 2],  label="equiv noise", linestyle="dashed", color="blue")
        ax.legend()
        ax.set(xlabel="time", ylabel="amplitude")
        ax.set_title("%d step"%past_x0_coeff.shape[0])
        plt.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close()
    return


class CAnalyzer:
    def __init__(self):
        self.expr_pool = {}
        return

    def add_item(self, key, val):
        assert key not in self.expr_pool
        self.expr_pool[key] = val
        return

    def get_item(self, key):
        assert key in self.expr_pool
        return self.expr_pool[key]

    def get_y_symbols(self):
        y_symbols = []
        for key, val in self.expr_pool.items():
            if key.startswith("y_"):
                assert(isinstance(val, sympy.core.symbol.Symbol))
                y_symbols.append(val)
        return y_symbols

    def get_eps_symbols(self):
        eps_symbols = []
        for key, val in self.expr_pool.items():
            if key.startswith("eps_"):
                assert(isinstance(val, sympy.core.symbol.Symbol))
                eps_symbols.append(val)
        return eps_symbols

    def show_symbol_coeff(self, expr, symbols):
        coeffs = []
        for symbol in symbols:
            coeff = float(expr.coeff(symbol))
            coeffs.append(coeff)
            # if not np.isclose(coeff, 0):
            #     print(symbol.name, coeff)
        return np.array(coeffs)