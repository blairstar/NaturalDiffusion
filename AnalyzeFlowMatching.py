import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scienceplots


from Utils import draw_marginal_coeff


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
        arr_eps[end - start - 1, 0] = eps

        xzs = []
        for ii in range(start, end)[::-1]:
            base = float(coeff_x0[ii])
            factor = float(np.prod(coeff_xt[start:ii]))
            xzs.append(base * factor)
        arr_xz[end - start - 1, :end - start] = np.array(xzs)

        o2 = eps
        o1 = np.array(xzs).sum()

        g2 = sigmas[start]
        g1 = 1 - sigmas[start]
        node_coeff[end - start - 1, :] = np.array([sigmas[start], g1, g2])

        print("start", start)
        print("pred: %0.4f %0.4f" % (o1, o2))
        print("true: %0.4f %0.4f" % (g1, g2))

    names = ["%0.3f" % node_coeff[ii, 0] for ii in range(0, num_step)]
    df = pd.DataFrame(arr_xz.round(3), columns=names, index=names)
    df["sum"] = arr_xz.sum(axis=1).round(3)
    df.to_csv("results/flow_euler/flow_euler_%03d.csv" % num_step)
    print(df)

    node_coeff = np.vstack([np.array([1.0, 0.0, 1.0]), node_coeff])

    draw_marginal_coeff(arr_xz, arr_eps, node_coeff, "results/flow_euler/flow_euler_%03d.jpg" % num_step)

    np.savez("results/flow_euler/flow_euler_%03d.npz" % num_step, past_x0_coeff=arr_xz, past_eps_coeff=arr_eps,
             node_coeff=node_coeff)
    print(arr_xz)
    print(arr_eps)

    return


def flow_analyze_coeff_tx():
    for num_step in [18, 24, 100, 500]:
        flow_analyze_coeff(num_step)
    return


if __name__ == "__main__":
    flow_analyze_coeff_tx()
