# # # most of the codes are copy from https://github.com/yang-song/score_sde.git

import abc
import numpy as np
import torch
import sympy
from sympy import symbols


expr_pool = {}


def batch_mul(a, b):
    return a*b


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a JAX tensor.
          t: a JAX float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * np.sqrt(dt)
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - batch_mul(diffusion ** 2, score * (0.5 if self.probability_flow else 1.))
                # Set the diffusion function to zero for ODEs.
                diffusion = np.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - batch_mul(G ** 2, score_fn(x, t) * (0.5 if self.probability_flow else 1.))
                rev_G = np.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


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
    
    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a JAX tensor.
          t: a JAX float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * np.sqrt(dt)
        return f, G


class ReverseSDE:
    def __init__(self, forward_sde, probability_flow, score_fn):
        self.forward_sde = forward_sde
        self.N = self.forward_sde.N
        self.probability_flow = probability_flow
        self.score_fn = score_fn
        return

    def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        forward_drift, forward_diffusion = self.forward_sde.sde(x, t)
        score = self.score_fn(x, t)
        reverse_drift = forward_drift - batch_mul(forward_diffusion ** 2, score * (0.5 if self.probability_flow else 1.))
        # Set the diffusion function to zero for ODEs.
        reverse_diffusion = np.zeros_like(forward_diffusion) if self.probability_flow else forward_diffusion
        return reverse_drift, reverse_diffusion 

    

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, rng, x, t):
        """One update of the predictor.

        Args:
          rng: A JAX random state.
          x: A JAX array representing the current state
          t: A JAX array representing the current time step.

        Returns:
          x: A JAX array of the next state.
          x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, rng, x, t):
        dt = -1. / self.rsde.N
        z = random.normal(rng, x.shape)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + batch_mul(diffusion, np.sqrt(-dt) * z)
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, rng, x, t):
        f, G = self.rsde.discretize(x, t)
        z = random.normal(rng, x.shape)
        x_mean = x - f
        x = x_mean + batch_mul(G, z)
        return x, x_mean


def add_item(key, val):
    assert key not in expr_pool
    expr_pool[key] = val
    return expr_pool


def get_item(key):
    assert key in expr_pool
    return expr_pool[key]


def get_y_symbols():
    y_symbols = []
    for key, val in expr_pool.items():
        if key.startswith("y_"):
            assert (isinstance(val, sympy.core.symbol.Symbol))
            y_symbols.append(val)
    return y_symbols


def get_eps_symbols():
    eps_symbols = []
    for key, val in expr_pool.items():
        if key.startswith("eps_"):
            assert (isinstance(val, sympy.core.symbol.Symbol))
            eps_symbols.append(val)
    return eps_symbols


def show_symbol_coeff(expr, symbols):
    coeffs = []
    for symbol in symbols:
        coeff = float(expr.coeff(symbol))
        coeffs.append(coeff)
        if not np.isclose(coeff, 0):
            print(symbol.name, coeff)
    return np.array(coeffs)



def score(xt, pred_x0, sigma):
    return (pred_x0-xt)/sigma**2


def sampling_ode_tx():
    N = 50
    eta = 1/N
    sde = VPSDE(N=N)
    
    time_nodes = 1 + np.arange(0, N, 1)*(eta-1)/(N-1)
    
    time_nodes = torch.from_numpy(time_nodes)

    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)
    
    all_time_nodes = [time_nodes[0]]
    for ii in range(N-1):
        s = time_nodes[ii]
        t = time_nodes[ii+1]

        dt = (eta-1)/(N-1)
        x_s = get_item("x_%0.4f" % s)
        y_s = symbols("y_%0.4f" % s)
        add_item("y_%0.4f" % s, y_s)
        
        y_coeff, noise_coeff = sde.marginal_prob_coeff(s)
        score = (y_coeff*y_s - x_s)/noise_coeff**2

        fn_coeff, gn_coeff = sde.sde_coeff(s)
        velocity = fn_coeff*x_s - 0.5 * gn_coeff**2 * score
        x_t = x_s + velocity*dt
        if ii+1 == N-1:
            print("a")
            
        add_item("x_%0.4f" % t, x_t)
        all_time_nodes.append(t)
        
    all_time_nodes = sorted(list(set(all_time_nodes)), reverse=True)

    ys = get_y_symbols()
    epss = get_eps_symbols()

    for t in all_time_nodes:
        x_t = get_item("x_%0.4f" % t)

        true_y_alpha, true_eps_sigma = sde.marginal_prob_coeff(t)
        
        print("t", t)
        y_coeffs = show_symbol_coeff(x_t, ys)
        print("y result", np.sum(y_coeffs), true_y_alpha)

        eps_coeffs = show_symbol_coeff(x_t, epss)
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_sigma)
        print("")

    return


def sampling_sde_tx():
    N = 20
    eta = 1 / N
    sde = VPSDE(N=N)

    time_nodes = 1 + np.arange(0, N, 1) * (eta - 1) / (N - 1)

    time_nodes = torch.from_numpy(time_nodes)

    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)

    all_time_nodes = [time_nodes[0]]
    for ii in range(N - 1):
        s = time_nodes[ii]
        t = time_nodes[ii + 1]

        dt = (eta - 1) / (N - 1)
        x_s = get_item("x_%0.4f" % s)
        y_s = symbols("y_%0.4f" % s)
        add_item("y_%0.4f" % s, y_s)

        y_coeff, noise_coeff = sde.marginal_prob_coeff(s)
        score = (y_coeff*y_s - x_s) / noise_coeff**2

        fn_coeff, gn_coeff = sde.sde_coeff(s)
        velocity = fn_coeff*x_s - gn_coeff**2 * score
        
        eps_t = symbols("eps_%0.4f"%t)
        add_item("eps_%0.4f"%t, eps_t)
        noise_scale = gn_coeff * np.sqrt(np.abs(dt))
        
        x_t = x_s + velocity*dt + noise_scale*eps_t
        if ii + 1 == N - 1:
            print("a")

        add_item("x_%0.4f" % t, x_t)
        all_time_nodes.append(t)

    all_time_nodes = sorted(list(set(all_time_nodes)), reverse=True)

    ys = get_y_symbols()
    epss = get_eps_symbols()

    for t in all_time_nodes:
        x_t = get_item("x_%0.4f" % t)

        true_y_alpha, true_eps_sigma = sde.marginal_prob_coeff(t)

        print("t", t)
        y_coeffs = show_symbol_coeff(x_t, ys)
        print("y result", np.sum(y_coeffs), true_y_alpha)

        eps_coeffs = show_symbol_coeff(x_t, epss)
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_sigma)
        print("")

    return


def sampling_heun_tx():
    N = 100
    eta = 1 / N
    sde = VPSDE(N=N)

    time_nodes = 1 + np.arange(0, N, 1) * (eta - 1) / (N - 1)

    time_nodes = torch.from_numpy(time_nodes)

    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)

    all_time_nodes = [time_nodes[0]]
    for ii in range(0, N-1, 1):
        s = time_nodes[ii]
        t = time_nodes[ii+1]

        dt = (eta-1)/(N - 1)
        
        # first step
        x_s = get_item("x_%0.4f" % s)
        y_s = symbols("y_%0.4f_org" % s)
        add_item("y_%0.4f_org" % s, y_s)

        y_coeff_s, noise_coeff_s = sde.marginal_prob_coeff(s)
        score_s = (y_coeff_s*y_s - x_s) / noise_coeff_s**2

        fn_coeff_s, gn_coeff_s = sde.sde_coeff(s)
        velocity_s = fn_coeff_s*x_s - 0.5 * gn_coeff_s**2 * score_s
        x_t_hat = x_s + velocity_s * dt
        
        # second step
        y_t = symbols("y_%0.4f_hat" % t)
        add_item("y_%0.4f_hat" % t, y_t)
        
        y_coeff_t, noise_coeff_t = sde.marginal_prob_coeff(t)
        score_t = (y_coeff_s*y_t - x_t_hat) / noise_coeff_t**2

        fn_coeff_t, gn_coeff_t = sde.sde_coeff(t)
        velocity_t = fn_coeff_t*x_t_hat - 0.5 * gn_coeff_t**2 * score_t

        x_t = x_s + 0.5*(velocity_s + velocity_t) * dt

        add_item("x_%0.4f" % t, x_t)
        all_time_nodes.append(t)

    all_time_nodes = sorted(list(set(all_time_nodes)), reverse=True)

    ys = get_y_symbols()
    epss = get_eps_symbols()

    for t in all_time_nodes:
        x_t = get_item("x_%0.4f" % t)

        true_y_alpha, true_eps_sigma = sde.marginal_prob_coeff(t)

        print("t", t)
        y_coeffs = show_symbol_coeff(x_t, ys)
        print("y result", np.sum(y_coeffs), true_y_alpha)

        eps_coeffs = show_symbol_coeff(x_t, epss)
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_sigma)
        print("")

    return


if __name__ == "__main__":
    # sampling_sde_tx()
    # sampling_ode_tx()
    sampling_heun_tx()