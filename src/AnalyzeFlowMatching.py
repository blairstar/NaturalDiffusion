
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
import os

from sympy import symbols

from Utils import CAnalyzer, draw_marginal_coeff, save_coeff_matrix


np.set_printoptions(suppress=True, linewidth=200, precision=3)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


def flow_analyze_coeff(num_step=50):
    sigmas = np.linspace(0, 1, num_step + 1)
    coeff_x0 = 1 - sigmas[:-1] / sigmas[1:]
    coeff_xt = sigmas[:-1] / sigmas[1:]

    arr_eps, arr_xz = np.zeros([num_step, num_step + 1]), np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step, 3])

    end = num_step
    for start in range(0, end, 1):
        eps = np.prod(coeff_xt[start:end])
        arr_eps[end-start-1, 0] = eps

        xzs = []
        for ii in range(start, end)[::-1]:
            base = float(coeff_x0[ii])
            factor = float(np.prod(coeff_xt[start:ii]))
            xzs.append(base * factor)
        arr_xz[end-start-1, :end-start] = np.array(xzs)

        o2 = eps
        o1 = np.array(xzs).sum()

        g2 = sigmas[start]
        g1 = 1 - sigmas[start]
        node_coeff[end-start-1, :] = np.array([sigmas[start], g1, g2])

        print("start", start)
        print("pred: %0.4f %0.4f" % (o1, o2))
        print("true: %0.4f %0.4f" % (g1, g2))

    node_coeff = np.vstack([np.array([1.0, 0.0, 1.0]), node_coeff])
    
    save_coeff_matrix(arr_xz, arr_eps, node_coeff, root_path/"results/flow_euler", "flow_euler")
    
    print(arr_xz)
    print(arr_eps)
    print(node_coeff)

    return


def flow_simpy_analyze_coeff(num_step):
    analyzer = CAnalyzer()

    time_nodes = np.linspace(0, 1, num_step+1)
    time_nodes = time_nodes[::-1]

    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    analyzer.add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    analyzer.add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)

    for ii in range(num_step):
        s = time_nodes[ii]
        t = time_nodes[ii+1]

        dt = t - s
        x_s = analyzer.get_item("x_%0.4f" % s)
        y_s = symbols("y_%0.4f" % s)
        analyzer.add_item("y_%0.4f"%s, y_s)

        velocity = (x_s - y_s)/s
        x_t = x_s + velocity * dt

        analyzer.add_item("x_%0.4f" % t, x_t)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()

    past_xstart_coeff = np.zeros([num_step, num_step])
    past_epsilon_coeff = np.zeros([num_step, num_step+1])
    node_coeff = np.zeros([num_step+1, 3])

    for kk, t in enumerate(time_nodes):
        x_t = analyzer.get_item("x_%0.4f" % t)

        true_y_alpha, true_eps_sigma = 1 - t, t

        print("t", t)
        y_coeffs = analyzer.show_symbol_coeff(x_t, ys)
        print("y result", np.sum(y_coeffs), true_y_alpha)

        eps_coeffs = analyzer.show_symbol_coeff(x_t, epss)
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_sigma)
        print("")

        node_coeff[kk, :] = np.array([t, true_y_alpha, true_eps_sigma])
        if not np.isclose(t, 1.0):
            past_xstart_coeff[kk - 1, :len(y_coeffs)] = np.array(y_coeffs)
            past_epsilon_coeff[kk - 1, :len(eps_coeffs)] = np.array(eps_coeffs)

    print(past_xstart_coeff)
    print(past_epsilon_coeff)
    print(node_coeff)

    save_coeff_matrix(past_xstart_coeff, past_epsilon_coeff, node_coeff, root_path/"results/flow_euler", "flow_euler_simpy")


def flow_analyze_coeff_tx():
    for num_step in [18, 24, 100, 500]:
        flow_analyze_coeff(num_step)
        break
    return


def flow_simpy_analyze_coeff_tx():
    for num_step in [18, 24, 100, 200]:
        flow_simpy_analyze_coeff(num_step)
    return


if __name__ == "__main__":
    '''
    Here, we offer two options: one is to compute directly through the analytical expression, and the other is to leverage SymPy for automatic computation.
    For SimPy, when the number of steps exceeds 200, the computation becomes relatively slow.
    '''
    # flow_analyze_coeff_tx()
    flow_simpy_analyze_coeff_tx()
