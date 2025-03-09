import numpy as np
from sympy import symbols
import sympy
import torch

expr_pool = {}


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def create_sde_coeff(skip_step=10):
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    std = np.sqrt(posterior_variance)

    coeff_x0 = betas*np.sqrt(alphas_bar)/(1-alphas_bar)
    coeff_xt = 2-np.sqrt(1-betas) - betas/(1-alphas_bar)

    skip_alphas_bar = alphas_bar[::skip_step]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii - 1]

    skip_betas = 1 - skip_alphas

    skip_std = np.sqrt(skip_betas)

    skip_coeff_x0 = skip_betas * np.sqrt(skip_alphas_bar) / (1 - skip_alphas_bar)
    skip_coeff_xt = (2 - np.sqrt(1 - skip_betas) - skip_betas / (1 - skip_alphas_bar))

    skip_coeff = skip_alphas, skip_alphas_bar, skip_std, skip_coeff_x0, skip_coeff_xt
    coeff = alphas, alphas_bar, std, coeff_x0, coeff_xt

    return skip_coeff, coeff


def sde_equivalent_coeff_tx():
    skip = 1
    skip_coeff, coeff = create_sde_coeff(skip)
    alphas, alphas_bar, std, coeff_x0, coeff_xt = coeff
    skip_alphas, skip_alphas_bar, skip_std, skip_coeff_x0, skip_coeff_xt = skip_coeff

    end = 1000
    for start in range(0, end, 10):
        epss = [np.prod(skip_coeff_xt[start:end])]
        for ii in range(start, end)[::-1]:
            sigma = float(skip_std[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            epss.append(sigma * factor)

        xzs = []
        for ii in range(start, end)[::-1]:
            base = float(skip_coeff_x0[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            xzs.append(base * factor)

        o2 = np.linalg.norm(np.array(epss))
        o1 = np.array(xzs).sum()

        g2 = np.sqrt(1 - alphas_bar[start * skip])
        g1 = np.sqrt(alphas_bar[start * skip])

        print("start", start)
        print("pred", o1, o2)
        print("true", g1, g2)

    return


def create_ode_coeff(skip_step=10):
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    coeff_x0 = np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar)
    coeff_xt = np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar)

    skip_alphas_bar = alphas_bar[::skip_step]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii - 1]

    skip_betas = 1 - skip_alphas

    skip_coeff_x0 = 0.5 * skip_betas * np.sqrt(skip_alphas_bar) / (1 - skip_alphas_bar)
    skip_coeff_xt = (2 - np.sqrt(1 - skip_betas) - 0.5 * skip_betas / (1 - skip_alphas_bar))

    skip_coeff = skip_alphas, skip_alphas_bar, skip_coeff_x0, skip_coeff_xt
    coeff = alphas, alphas_bar, coeff_x0, coeff_xt

    return skip_coeff, coeff


def ode_equivalent_coeff_tx():
    skip = 1
    skip_coeff, coeff = create_ode_coeff(skip)
    alphas, alphas_bar, coeff_x0, coeff_xt = coeff
    skip_alphas, skip_alphas_bar, skip_coeff_x0, skip_coeff_xt = skip_coeff

    end = 1000
    for start in range(0, end, 10):
        eps = np.prod(skip_coeff_xt[start:end])

        xzs = []
        for ii in range(start, end)[::-1]:
            base = float(skip_coeff_x0[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            xzs.append(base * factor)

        o2 = eps
        o1 = np.array(xzs).sum()

        g2 = np.sqrt(1 - alphas_bar[start * skip])
        g1 = np.sqrt(alphas_bar[start * skip])

        print("start", start)
        print("pred", o1, o2)
        print("true", g1, g2)

    return

    
if __name__ == "__main__":
    # ode_equivalent_coeff_tx()
    sde_equivalent_coeff_tx()
    