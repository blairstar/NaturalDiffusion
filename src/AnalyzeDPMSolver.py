# # Part of the code is copied from https://github.com/LuChengTHU/dpm-solver

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path/"src"))

import numpy as np
from sympy import symbols
import torch
import pandas as pd

from Utils import CAnalyzer, save_coeff_matrix

np.set_printoptions(suppress=True, linewidth=200, precision=3)


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand



class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support the linear VPSDE for the continuous time setting. The hyperparameters for the noise
            schedule are the default settings in Yang Song's ScoreSDE:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear']:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues. 
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device),
                                  self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                               torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))


def analyze_dpmsolver_2s(step=15):
    analyzer = CAnalyzer()
    
    ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20)

    time_nodes = np.linspace(1.0, 0.001, step+1)
    time_nodes = torch.from_numpy(time_nodes)
    total_step = 2*step
    
    eps_t = symbols("eps_%0.4f"%(time_nodes[0]))
    analyzer.add_item("eps_%0.4f"%(time_nodes[0]), eps_t)
    analyzer.add_item("x_%0.4f"%time_nodes[0], eps_t*1.0)
    
    all_time_nodes = []
    for ii in range(step):
        s = time_nodes[ii]
        t = time_nodes[ii+1]
        
        r1 = 0.5
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        
        s1 = ns.inverse_lambda(lambda_s1)
        all_time_nodes.extend([s.item(), s1.item(), t.item()])
        
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s, alpha_s1, alpha_t = torch.exp(log_alpha_s), torch.exp(log_alpha_s1), torch.exp(log_alpha_t)
         
        x_s = analyzer.get_item("x_%0.4f"%s)
        
        # first step
        x_s1_s_coeff = torch.exp(log_alpha_s1 - log_alpha_s).item()
        
        phi_s1_s = torch.expm1(r1 * h)
        eps_s1_s_coeff = (sigma_s1*phi_s1_s).item()

        y_s = symbols("y_%0.4f"%s)
        analyzer.add_item("y_%0.4f"%s, y_s)
        
        model_s = (x_s - alpha_s*y_s)/sigma_s
        
        x_s1 = x_s1_s_coeff*x_s - eps_s1_s_coeff*model_s
        
        # second step
        x_t_s_coeff = torch.exp(log_alpha_t - log_alpha_s).item()
        
        y_s1 = symbols("y_%0.4f"%s1)
        analyzer.add_item("y_%0.4f"%s1, y_s1)
        model_s1 = (x_s1 - alpha_s1*y_s1)/sigma_s1
        
        phi_t_s = torch.expm1(h)
        eps_t_s_coeff = (sigma_t*phi_t_s).item()
        eps_diff_t_s_coeff = ((0.5/r1)*sigma_t*phi_t_s).item()

        x_t = x_t_s_coeff*x_s - eps_t_s_coeff*model_s - eps_diff_t_s_coeff*(model_s1 - model_s)

        analyzer.add_item("x_%0.4f"%s1, x_s1)
        analyzer.add_item("x_%0.4f"%t, x_t)
    
    # all_time_nodes = sorted(list(set(all_time_nodes)), reverse=True)
    all_time_nodes = sorted(list(np.unique(np.array(all_time_nodes))), reverse=True)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()
    
    past_xstart_coeff = np.zeros([total_step, total_step])
    past_epsilon_coeff = np.zeros([total_step, total_step+1])
    node_coeff = np.zeros([total_step+1, 3])

    for kk, t in enumerate(all_time_nodes):
        x_t = analyzer.get_item("x_%0.4f"%t)
        
        print("t", t)
        y_coeffs = analyzer.show_symbol_coeff(x_t, ys)
        true_y_alpha = ns.marginal_alpha(torch.tensor(t)).item()
        print("y result", np.sum(y_coeffs), true_y_alpha)
        
        eps_coeffs = analyzer.show_symbol_coeff(x_t, epss)
        true_eps_sigma = ns.marginal_std(torch.tensor(t)).item()
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_sigma)
        print("")

        node_coeff[kk, :] = np.array([t, true_y_alpha, true_eps_sigma])
        if not np.isclose(t, 1.0):
            past_xstart_coeff[kk-1, :len(y_coeffs)] = np.array(y_coeffs)
            past_epsilon_coeff[kk-1, :len(eps_coeffs)] = np.array(eps_coeffs)

    save_coeff_matrix(past_xstart_coeff, past_epsilon_coeff, node_coeff, root_path/"results/dpmsolver", "dpmsolver2s")
    
    print(past_xstart_coeff)
    print(past_epsilon_coeff)
    print(node_coeff) 
        
    return


def analyze_dpmsolver_pp_2s(step=15):
    analyzer = CAnalyzer()
    
    ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20)

    time_nodes = np.linspace(1.0, 0.001, step + 1)
    time_nodes = torch.from_numpy(time_nodes)
    total_step = 2*step

    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    analyzer.add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    analyzer.add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)

    all_time_nodes = []
    for ii in range(step):
        s = time_nodes[ii]
        t = time_nodes[ii + 1]

        r1 = 0.5
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h

        s1 = ns.inverse_lambda(lambda_s1)
        all_time_nodes.extend([s.item(), s1.item(), t.item()])

        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_t = ns.marginal_log_mean_coeff(t)

        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s, alpha_s1, alpha_t = torch.exp(log_alpha_s), torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        x_s = analyzer.get_item("x_%0.4f" % s)

        # first step
        x_s1_s_coeff = (sigma_s1/sigma_s).item()

        phi_s1_s = torch.expm1(-r1 * h)
        eps_s1_s_coeff = (alpha_s1 * phi_s1_s).item()

        y_s = symbols("y_%0.4f" % s)
        analyzer.add_item("y_%0.4f" % s, y_s)

        # model_s = (x_s - alpha_s * y_s) / sigma_s
        model_s = y_s
        
        x_s1 = x_s1_s_coeff * x_s - eps_s1_s_coeff * model_s

        # second step
        x_t_s_coeff = (sigma_t/sigma_s).item()

        y_s1 = symbols("y_%0.4f" % s1)
        analyzer.add_item("y_%0.4f" % s1, y_s1)
        # model_s1 = (x_s1 - alpha_s1 * y_s1) / sigma_s1
        model_s1 = y_s1
        
        phi_t_s = torch.expm1(-h)
        eps_t_s_coeff = (alpha_t * phi_t_s).item()
        eps_diff_t_s_coeff = ((0.5 / r1) * alpha_t * phi_t_s).item()

        x_t = x_t_s_coeff * x_s - eps_t_s_coeff * model_s - eps_diff_t_s_coeff * (model_s1 - model_s)

        analyzer.add_item("x_%0.4f" % s1, x_s1)
        analyzer.add_item("x_%0.4f" % t, x_t)

    all_time_nodes = sorted(list(np.unique(np.array(all_time_nodes))), reverse=True)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()
     
    past_xstart_coeff = np.zeros([total_step, total_step])
    past_epsilon_coeff = np.zeros([total_step, total_step+1])
    node_coeff = np.zeros([total_step+1, 3])

    for kk, t in enumerate(all_time_nodes):
        x_t = analyzer.get_item("x_%0.4f" % t)

        print("t", t)
        y_coeffs = analyzer.show_symbol_coeff(x_t, ys)
        true_y_alpha = ns.marginal_alpha(torch.tensor(t)).item()
        print("y result", np.sum(y_coeffs), true_y_alpha)

        eps_coeffs = analyzer.show_symbol_coeff(x_t, epss)
        true_eps_sigma = ns.marginal_std(torch.tensor(t)).item()
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_sigma)
        print("")
        
        node_coeff[kk, :] = np.array([t, true_y_alpha, true_eps_sigma])
        if not np.isclose(t, 1.0):
            past_xstart_coeff[kk-1, :len(y_coeffs)] = np.array(y_coeffs)
            past_epsilon_coeff[kk-1, :len(eps_coeffs)] = np.array(eps_coeffs)

    save_coeff_matrix(past_xstart_coeff, past_epsilon_coeff, node_coeff, root_path/"results/dpmsolverpp", "dpmsolverpp2s")
    
    print(past_xstart_coeff)
    print(past_epsilon_coeff)
    print(node_coeff)
     
    return


def analyze_dpmsolver_3s(step=10):
    analyzer = CAnalyzer()

    ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20)

    time_nodes = np.linspace(1.0, 0.001, step + 1)
    time_nodes = torch.from_numpy(time_nodes)
    total_step = 3*step
    
    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    analyzer.add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    analyzer.add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)

    all_time_nodes = []
    for ii in range(step):
        s = time_nodes[ii]
        t = time_nodes[ii + 1]

        r1, r2 = 1/3, 2/3
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1*h
        lambda_s2 = lambda_s + r2*h

        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        all_time_nodes.extend([s.item(), s1.item(), s2.item(), t.item()])

        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_s2 = ns.marginal_log_mean_coeff(s2)
        log_alpha_t = ns.marginal_log_mean_coeff(t)

        sigma_s, sigma_s1 = ns.marginal_std(s), ns.marginal_std(s1)
        sigma_s2, sigma_t = ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s, alpha_s1 = torch.exp(log_alpha_s), torch.exp(log_alpha_s1)
        alpha_s2, alpha_t = torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        x_s = analyzer.get_item("x_%0.4f" % s)
        
        # first step
        phi_s1_s = torch.expm1(r1 * h)
        x_s1_s_coeff = torch.exp(log_alpha_s1 - log_alpha_s).item()
        eps_s1_s_coeff = (sigma_s1 * phi_s1_s).item()

        y_s = symbols("y_%0.4f" % s)
        analyzer.add_item("y_%0.4f" % s, y_s)
        model_s = (x_s - alpha_s * y_s) / sigma_s

        x_s1 = x_s1_s_coeff * x_s - eps_s1_s_coeff * model_s

        # second step
        x_s2_s_coeff = torch.exp(log_alpha_s2 - log_alpha_s).item()

        phi_s2_s_eps = torch.expm1(r2 * h)
        phi_s2_s_eps_diff = torch.expm1(r2 * h) / (r2 * h) - 1.
        eps_s2_s_coeff = (sigma_s2 * phi_s2_s_eps).item()
        eps_diff_s2_s_coeff = ((r2/r1) * sigma_s2 * phi_s2_s_eps_diff).item()
        
        y_s1 = symbols("y_%0.4f" % s1)
        analyzer.add_item("y_%0.4f" % s1, y_s1)
        model_s1 = (x_s1 - alpha_s1*y_s1) / sigma_s1

        x_s2 = x_s2_s_coeff*x_s - eps_s2_s_coeff*model_s - eps_diff_s2_s_coeff*(model_s1 - model_s)
        
        # third step
        x_t_s_coeff = torch.exp(log_alpha_t - log_alpha_s).item()
        
        phi_t_s_eps = torch.expm1(h)
        phi_t_s_eps_diff = phi_t_s_eps / h - 1.
        eps_t_s_coeff = (sigma_t * phi_t_s_eps).item()
        eps_diff_t_s_coeff = ((1/r2) * sigma_t * phi_t_s_eps_diff).item()
        
        y_s2 = symbols("y_%0.4f" % s2)
        analyzer.add_item("y_%0.4f" % s2, y_s2)
        model_s2 = (x_s2 - alpha_s2*y_s2) / sigma_s2

        x_t = x_t_s_coeff*x_s - eps_t_s_coeff*model_s - eps_diff_t_s_coeff*(model_s2 - model_s)
        
        analyzer.add_item("x_%0.4f" % s1, x_s1)
        analyzer.add_item("x_%0.4f" % s2, x_s2)
        analyzer.add_item("x_%0.4f" % t, x_t)

    all_time_nodes = sorted(list(np.unique(np.array(all_time_nodes))), reverse=True)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()

    past_xstart_coeff = np.zeros([total_step, total_step])
    past_epsilon_coeff = np.zeros([total_step, total_step+1])
    node_coeff = np.zeros([total_step+1, 3])
    
    for kk, t in enumerate(all_time_nodes):
        x_t = analyzer.get_item("x_%0.4f" % t)

        print("t", t)
        y_coeffs = analyzer.show_symbol_coeff(x_t, ys)
        true_y_alpha = ns.marginal_alpha(torch.tensor(t)).item()
        print("y result", np.sum(y_coeffs), true_y_alpha)

        eps_coeffs = analyzer.show_symbol_coeff(x_t, epss)
        true_eps_sigma = ns.marginal_std(torch.tensor(t)).item()
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_sigma)
        print("")
        
        node_coeff[kk, :] = np.array([t, true_y_alpha, true_eps_sigma])
        if not np.isclose(t, 1.0):
            past_xstart_coeff[kk-1, :len(y_coeffs)] = np.array(y_coeffs)
            past_epsilon_coeff[kk-1, :len(eps_coeffs)] = np.array(eps_coeffs)

    save_coeff_matrix(past_xstart_coeff, past_epsilon_coeff, node_coeff, root_path/"results/dpmsolver", "dpmsolver3s")
    
    print(past_xstart_coeff)
    print(past_epsilon_coeff)
    print(node_coeff)
 
    return


def analyze_dpmsolver_pp_3s(step=10):
    analyzer = CAnalyzer()

    ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20)

    time_nodes = np.linspace(1.0, 0.001, step + 1)
    time_nodes = torch.from_numpy(time_nodes)
    total_step = 3*step
    
    eps_t = symbols("eps_%0.4f" % (time_nodes[0]))
    analyzer.add_item("eps_%0.4f" % (time_nodes[0]), eps_t)
    analyzer.add_item("x_%0.4f" % time_nodes[0], eps_t * 1.0)

    all_time_nodes = []
    for ii in range(step):
        s = time_nodes[ii]
        t = time_nodes[ii + 1]

        r1, r2 = 1/3, 2/3
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h

        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        all_time_nodes.extend([s.item(), s1.item(), s2.item(), t.item()])

        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_s2 = ns.marginal_log_mean_coeff(s2)
        log_alpha_t = ns.marginal_log_mean_coeff(t)

        sigma_s, sigma_s1 = ns.marginal_std(s), ns.marginal_std(s1)
        sigma_s2, sigma_t = ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s, alpha_s1 = torch.exp(log_alpha_s), torch.exp(log_alpha_s1)
        alpha_s2, alpha_t = torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        x_s = analyzer.get_item("x_%0.4f" % s)

        # first step
        phi_s1_s = torch.expm1(-r1*h)
        x_s1_s_coeff = (sigma_s1/sigma_s).item()
        eps_s1_s_coeff = (alpha_s1 * phi_s1_s).item()

        y_s = symbols("y_%0.4f" % s)
        analyzer.add_item("y_%0.4f"%s, y_s)
        model_s = y_s

        x_s1 = x_s1_s_coeff*x_s - eps_s1_s_coeff*model_s

        # second step
        x_s2_s_coeff = (sigma_s2/sigma_s).item()

        phi_s2_s_eps = torch.expm1(-r2*h)
        phi_s2_s_eps_diff = torch.expm1(-r2*h)/(r2*h) + 1.0
        eps_s2_s_coeff = (alpha_s2 * phi_s2_s_eps).item()
        eps_diff_s2_s_coeff = ((r2 / r1) * alpha_s2 * phi_s2_s_eps_diff).item()

        y_s1 = symbols("y_%0.4f" % s1)
        analyzer.add_item("y_%0.4f" % s1, y_s1)
        model_s1 = y_s1

        x_s2 = x_s2_s_coeff*x_s - eps_s2_s_coeff*model_s - eps_diff_s2_s_coeff*(model_s1 - model_s)

        # third step
        x_t_s_coeff = (sigma_t/sigma_s).item()

        phi_t_s_eps = torch.expm1(-h)
        phi_t_s_eps_diff = phi_t_s_eps/h + 1.0
        eps_t_s_coeff = (alpha_t * phi_t_s_eps).item()
        eps_diff_t_s_coeff = ((1.0/r2) * alpha_t * phi_t_s_eps_diff).item()

        y_s2 = symbols("y_%0.4f" % s2)
        analyzer.add_item("y_%0.4f" % s2, y_s2)
        model_s2 = y_s2

        x_t = x_t_s_coeff*x_s - eps_t_s_coeff*model_s - eps_diff_t_s_coeff*(model_s2 - model_s)

        analyzer.add_item("x_%0.4f" % s1, x_s1)
        analyzer.add_item("x_%0.4f" % s2, x_s2)
        analyzer.add_item("x_%0.4f" % t, x_t)

    all_time_nodes = sorted(list(np.unique(np.array(all_time_nodes))), reverse=True)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()

    past_xstart_coeff = np.zeros([total_step, total_step])
    past_epsilon_coeff = np.zeros([total_step, total_step+1])
    node_coeff = np.zeros([total_step+1, 3])
    
    for kk, t in enumerate(all_time_nodes):
        x_t = analyzer.get_item("x_%0.4f" % t)

        print("t", t)
        y_coeffs = analyzer.show_symbol_coeff(x_t, ys)
        true_y_alpha = ns.marginal_alpha(torch.tensor(t)).item()
        print("y result", np.sum(y_coeffs), true_y_alpha)

        eps_coeffs = analyzer.show_symbol_coeff(x_t, epss)
        true_eps_sigma = ns.marginal_std(torch.tensor(t)).item()
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_sigma)
        print("")
        
        node_coeff[kk, :] = np.array([t, true_y_alpha, true_eps_sigma])
        if not np.isclose(t, 1.0):
            past_xstart_coeff[kk-1, :len(y_coeffs)] = np.array(y_coeffs)
            past_epsilon_coeff[kk-1, :len(eps_coeffs)] = np.array(eps_coeffs)

    save_coeff_matrix(past_xstart_coeff, past_epsilon_coeff, node_coeff, root_path/"results/dpmsolverpp", "dpmsolverpp3s")
    
    print(past_xstart_coeff)
    print(past_epsilon_coeff)
    print(node_coeff)
     
    return


def analyze_dpmsolver_2s_tx():
    for step in [9, 12, 50, 100]:
        analyze_dpmsolver_2s(step)
    return


def analyze_dpmsolver_pp_2s_tx():
    for step in [9, 12, 50, 100]:
        analyze_dpmsolver_pp_2s(step)
    return


def analyze_dpmsolver_3s_tx():
    for step in [6, 8, 33, 67]:
        analyze_dpmsolver_3s(step)
    return


def analyze_dpmsolver_pp_3s_tx():
    for step in [6, 8, 33, 67]:
        analyze_dpmsolver_pp_3s(step)
    return


if __name__ == "__main__":
    analyze_dpmsolver_2s_tx()
    analyze_dpmsolver_pp_2s_tx()
    analyze_dpmsolver_3s_tx()
    analyze_dpmsolver_pp_3s_tx()

    