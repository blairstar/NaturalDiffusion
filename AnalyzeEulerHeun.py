# # Part of the code is copied from https://github.com/yang-song/score_sde.git

import numpy as np
import torch
import sympy
from sympy import symbols
import pandas as pd

from Utils import draw_marginal_coeff, CAnalyzer


class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        return

    @property
    def T(self):
        return 1

    def sde_coeff(self, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        ft_coeff = -0.5*beta_t
        gt_coeff = np.sqrt(beta_t)
        return ft_coeff, gt_coeff

    def marginal_prob_coeff(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        xstart_coeff = np.exp(log_mean_coeff)
        noise_coeff = np.sqrt(1 - np.exp(2. * log_mean_coeff))
        return xstart_coeff, noise_coeff
    

def score(xt, pred_x0, sigma):
    return (pred_x0-xt)/sigma**2


def sampling_ode(N=50):
    analyzer = CAnalyzer()

    eta = 1/N
    sde = VPSDE(N=N)
    total_step = N-1

    time_nodes = 1 + np.arange(0, N, 1)*(eta-1)/(N-1)
    
    time_nodes = torch.from_numpy(time_nodes)

    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    analyzer.add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    analyzer.add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)
    
    all_time_nodes = [time_nodes[0]]
    for ii in range(N-1):
        s = time_nodes[ii]
        t = time_nodes[ii+1]

        dt = (eta-1)/(N-1)
        x_s = analyzer.get_item("x_%0.4f" % s)
        y_s = symbols("y_%0.4f" % s)
        analyzer.add_item("y_%0.4f" % s, y_s)
        
        y_coeff, noise_coeff = sde.marginal_prob_coeff(s)
        score = (y_coeff*y_s - x_s)/noise_coeff**2

        fn_coeff, gn_coeff = sde.sde_coeff(s)
        velocity = fn_coeff*x_s - 0.5 * gn_coeff**2 * score
        x_t = x_s + velocity*dt
        if ii+1 == N-1:
            print("a")
            
        analyzer.add_item("x_%0.4f" % t, x_t)
        all_time_nodes.append(t)
        
    all_time_nodes = sorted(list(set(all_time_nodes)), reverse=True)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()
    
    past_xstart_coeff = np.zeros([total_step, total_step])
    past_epsilon_coeff = np.zeros([total_step, total_step+1])
    node_coeff = np.zeros([total_step+1, 3])

    for kk, t in enumerate(all_time_nodes):
        x_t = analyzer.get_item("x_%0.4f" % t)

        true_y_alpha, true_eps_sigma = sde.marginal_prob_coeff(t)
        
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

    names = ["%0.3f" % node_coeff[ii + 1, 0] for ii in range(0, total_step)]
    df = pd.DataFrame(past_xstart_coeff.round(3), columns=names, index=names)
    df["sum"] = past_xstart_coeff.sum(axis=1).round(3)
    df.to_csv("results/euler_heun/ode_euler_%03d.csv" % total_step)
    print(df)

    draw_marginal_coeff(past_xstart_coeff, past_epsilon_coeff,
                        node_coeff, "results/euler_heun/ode_euler_%03d.jpg" % total_step)

    np.savez("results/euler_heun/ode_euler_%03d.npz" % total_step, past_xstart_coeff=past_xstart_coeff,
             past_epsilon_coeff=past_epsilon_coeff, node_coeff=node_coeff)

    return


def sampling_sde(N=50):
    analyzer = CAnalyzer()
    
    eta = 1 / N
    sde = VPSDE(N=N)
    total_step = N-1
    
    time_nodes = 1 + np.arange(0, N, 1) * (eta - 1) / (N - 1)

    time_nodes = torch.from_numpy(time_nodes)

    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    analyzer.add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    analyzer.add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)

    all_time_nodes = [time_nodes[0]]
    for ii in range(N - 1):
        s = time_nodes[ii]
        t = time_nodes[ii + 1]

        dt = (eta - 1) / (N - 1)
        x_s = analyzer.get_item("x_%0.4f" % s)
        y_s = symbols("y_%0.4f" % s)
        analyzer.add_item("y_%0.4f" % s, y_s)

        y_coeff, noise_coeff = sde.marginal_prob_coeff(s)
        score = (y_coeff*y_s - x_s) / noise_coeff**2

        fn_coeff, gn_coeff = sde.sde_coeff(s)
        velocity = fn_coeff*x_s - gn_coeff**2 * score
        
        eps_t = symbols("eps_%0.4f"%t)
        analyzer.add_item("eps_%0.4f"%t, eps_t)
        noise_scale = gn_coeff * np.sqrt(np.abs(dt))
        
        x_t = x_s + velocity*dt + noise_scale*eps_t

        analyzer.add_item("x_%0.4f" % t, x_t)
        all_time_nodes.append(t)

    all_time_nodes = sorted(list(set(all_time_nodes)), reverse=True)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()
    
    past_xstart_coeff = np.zeros([total_step, total_step])
    past_epsilon_coeff = np.zeros([total_step, total_step+1])
    node_coeff = np.zeros([total_step+1, 3])

    for kk, t in enumerate(all_time_nodes):
        x_t = analyzer.get_item("x_%0.4f" % t)

        true_y_alpha, true_eps_sigma = sde.marginal_prob_coeff(t)

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

    names = ["%0.3f" % node_coeff[ii + 1, 0] for ii in range(0, total_step)]
    df = pd.DataFrame(past_xstart_coeff.round(3), columns=names, index=names)
    df["sum"] = past_xstart_coeff.sum(axis=1).round(3)
    df.to_csv("results/euler_heun/sde_euler_%03d.csv" % total_step)
    print(df)

    draw_marginal_coeff(past_xstart_coeff, past_epsilon_coeff,
                        node_coeff, "results/euler_heun/sde_euler_%03d.jpg" % total_step)

    np.savez("results/euler_heun/sde_euler_%03d.npz" % total_step, past_xstart_coeff=past_xstart_coeff,
             past_epsilon_coeff=past_epsilon_coeff, node_coeff=node_coeff)
    return


def sampling_heun(N=25):
    analyzer = CAnalyzer()
    
    eta = 1 / N
    sde = VPSDE(N=N)
    offset = 0.0005
    
    time_nodes = 1 + np.arange(0, N, 1) * (eta - 1) / (N - 1)

    time_nodes = torch.from_numpy(time_nodes)

    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    analyzer.add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    analyzer.add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)
    
    total_step = 2*(N-1)

    all_time_nodes = [time_nodes[0]]
    for ii in range(0, N-1, 1):
        s = time_nodes[ii]
        t = time_nodes[ii+1]

        dt = (eta-1)/(N - 1)
        
        # first step
        x_s = analyzer.get_item("x_%0.4f" % s)
        y_s = symbols("y_%0.4f" % s)
        analyzer.add_item("y_%0.4f" % s, y_s)

        y_coeff_s, noise_coeff_s = sde.marginal_prob_coeff(s)
        score_s = (y_coeff_s*y_s - x_s) / noise_coeff_s**2

        fn_coeff_s, gn_coeff_s = sde.sde_coeff(s)
        velocity_s = fn_coeff_s*x_s - 0.5 * gn_coeff_s**2 * score_s
        x_t_hat = x_s + velocity_s * dt
       
        # For the Heun algorithm, there are repeated predictions at every t, so an offset is used for differentiation.
        analyzer.add_item("x_%0.4f"%(t+offset), x_t_hat)
        all_time_nodes.append(t+offset)

        # second step
        y_t_hat = symbols("y_%0.4f" %(t+offset))
        analyzer.add_item("y_%0.4f"%(t+offset), y_t_hat)
        
        y_coeff_t, noise_coeff_t = sde.marginal_prob_coeff(t)
        score_t = (y_coeff_s*y_t_hat - x_t_hat) / noise_coeff_t**2

        fn_coeff_t, gn_coeff_t = sde.sde_coeff(t)
        velocity_t = fn_coeff_t*x_t_hat - 0.5 * gn_coeff_t**2 * score_t

        x_t = x_s + 0.5*(velocity_s + velocity_t) * dt

        analyzer.add_item("x_%0.4f" % t, x_t)
        all_time_nodes.append(t)

    all_time_nodes = sorted(list(set(all_time_nodes)), reverse=True)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()
    
    past_xstart_coeff = np.zeros([total_step, total_step])
    past_epsilon_coeff = np.zeros([total_step, total_step+1])
    node_coeff = np.zeros([total_step+1, 3])
    
    for kk, t in enumerate(all_time_nodes):
        x_t = analyzer.get_item("x_%0.4f" % t)

        true_y_alpha, true_eps_sigma = sde.marginal_prob_coeff(t)

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

    names = ["%0.3f" % node_coeff[ii + 1, 0] for ii in range(0, total_step)]
    df = pd.DataFrame(past_xstart_coeff.round(3), columns=names, index=names)
    df["sum"] = past_xstart_coeff.sum(axis=1).round(3)
    df.to_csv("results/euler_heun/huen_%03d.csv" % total_step)
    print(df)

    draw_marginal_coeff(past_xstart_coeff, past_epsilon_coeff,
                        node_coeff, "results/euler_heun/heun_%03d.jpg" % total_step)

    np.savez("results/euler_heun/heun_%03d.npz" % total_step, past_xstart_coeff=past_xstart_coeff,
             past_epsilon_coeff=past_epsilon_coeff, node_coeff=node_coeff)
    return


def sampling_heun_tx():
    for step in [9, 12, 50, 100]:
        sampling_heun(step)
    return


def sampling_sde_tx():
    for step in [18, 24, 100, 200]:
        sampling_sde(step)
    return


def sampling_ode_tx():
    for step in [18, 24, 100, 200]:
        sampling_ode(step)
    return


if __name__ == "__main__":
    sampling_sde_tx()
    sampling_ode_tx()
    sampling_heun_tx()
    