
import numpy as np
import pandas as pd
import os

np.set_printoptions(suppress=True, linewidth=200, precision=3)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


def equal_coeff_tx():
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    std = np.sqrt(posterior_variance)

    coeff_x0 = np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar)
    coeff_xt = np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar)

    start, end = 950, 1000
    e1000 = coeff_xt[999] * coeff_xt[998] * coeff_xt[997] * coeff_xt[996]

    e999 = std[999] * coeff_xt[998] * coeff_xt[997] * coeff_xt[996]
    e998 = std[998] * coeff_xt[997] * coeff_xt[996]
    e997 = std[997] * coeff_xt[996]
    e996 = std[996]

    xz999 = coeff_x0[999] * coeff_xt[998] * coeff_xt[997] * coeff_xt[996]
    xz998 = coeff_x0[998] * coeff_xt[997] * coeff_xt[996]
    xz997 = coeff_x0[997] * coeff_xt[996]
    xz996 = coeff_x0[996]

    epss = [np.prod(coeff_xt[start:end])]
    for ii in range(start, end)[::-1]:
        sigma = float(std[ii])
        factor = float(np.prod(coeff_xt[start:ii]))
        epss.append(sigma * factor)

    xzs = []
    for ii in range(start, end)[::-1]:
        base = float(coeff_x0[ii])
        factor = float(np.prod(coeff_xt[start:ii]))
        xzs.append(base * factor)
    
    o2 = np.linalg.norm(np.array(epss))
    o1 = np.array(xzs).sum()
    
    g2 = np.sqrt(1-alphas_bar[start])
    g1 = np.sqrt(alphas_bar[start])
    
    print("start", start)
    print("pred", o1, o2)
    print("true", g1, g2)
    return


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


def create_skip_ddpm_coeff(skip_step=10):
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    std = np.sqrt(posterior_variance)

    coeff_x0 = np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar)
    coeff_xt = np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar)
    
    skip_alphas_bar = alphas_bar[::skip_step]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii]/skip_alphas_bar[ii-1]
    
    skip_betas = 1 - skip_alphas
    skip_alphas_bar_prev = np.append(1.0, skip_alphas_bar[:-1])

    skip_posterior_variance = skip_betas * (1.0 - skip_alphas_bar_prev) / (1.0 - skip_alphas_bar)
    skip_std = np.sqrt(skip_posterior_variance)

    skip_coeff_x0 = np.sqrt(skip_alphas_bar_prev) * skip_betas / (1 - skip_alphas_bar)
    skip_coeff_xt = np.sqrt(skip_alphas) * (1 - skip_alphas_bar_prev) / (1 - skip_alphas_bar)

    skip_coeff = skip_alphas, skip_alphas_bar, skip_std, skip_coeff_x0, skip_coeff_xt
    coeff = alphas, alphas_bar, std, coeff_x0, coeff_xt

    return skip_coeff, coeff


def ddpm_equivalent_coeff_tx():
    skip = 50
    skip_coeff, coeff = create_skip_ddpm_coeff(skip)
    alphas, alphas_bar, std, coeff_x0, coeff_xt = coeff
    skip_alphas, skip_alphas_bar, skip_std, skip_coeff_x0, skip_coeff_xt = skip_coeff

    num_step = int(1000/skip)
    arr_eps, arr_xz = np.zeros([num_step, num_step+1]), np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step, 2])
    
    end = num_step
    for start in range(0, end, 1):
        epss = [np.prod(skip_coeff_xt[start:end])]
        for ii in range(start, end)[::-1]:
            sigma = float(skip_std[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            epss.append(sigma * factor)
        arr_eps[end-start-1, :1+end-start] = np.array(epss)
        
        xzs = []
        for ii in range(start, end)[::-1]:
            base = float(skip_coeff_x0[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            xzs.append(base * factor)
        arr_xz[end-start-1, :end-start] = np.array(xzs)
        
        o2 = np.linalg.norm(np.array(epss))
        o1 = np.array(xzs).sum()

        g2 = np.sqrt(1 - alphas_bar[start*skip])
        g1 = np.sqrt(alphas_bar[start*skip])
        node_coeff[start, :] = np.array([g1, g2])
        
        print("start", start)
        print("pred", o1, o2)
        print("true", g1, g2)
    
    np.savez("ddpm_%03d.npz"%num_step, past_x0_coeff=arr_xz, past_eps_coeff=arr_eps, node_coeff=node_coeff)
    print(arr_xz)
    print(arr_eps)
    return


def create_skip_ddim_coeff(skip_step):
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    recified_factor = np.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar))

    coeff_x0 = np.sqrt(alphas_bar_prev) - recified_factor * np.sqrt(alphas_bar)
    coeff_xt = recified_factor

    skip_alphas_bar = alphas_bar[::skip_step]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii - 1]

    skip_betas = 1 - skip_alphas
    skip_alphas_bar_prev = np.append(1.0, skip_alphas_bar[:-1])
    
    skip_recified_factor = np.sqrt((1 - skip_alphas_bar_prev) / (1 - skip_alphas_bar))
    skip_coeff_x0 = np.sqrt(skip_alphas_bar_prev) - skip_recified_factor * np.sqrt(skip_alphas_bar)
    skip_coeff_xt = skip_recified_factor
    
    skip_coeff = skip_alphas, skip_alphas_bar, skip_coeff_x0, skip_coeff_xt
    coeff = alphas, alphas_bar, coeff_x0, coeff_xt
    
    return skip_coeff, coeff


def ddim_equivalent_coeff_tx():
    skip = 40
    skip_coeff, coeff = create_skip_ddim_coeff(skip)
    alphas, alphas_bar, coeff_x0, coeff_xt = coeff
    skip_alphas, skip_alphas_bar, skip_coeff_x0, skip_coeff_xt = skip_coeff

    num_step = int(1000 / skip)
    arr_eps, arr_xz = np.zeros([num_step, num_step + 1]), np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step, 2])

    end = num_step
    for start in range(0, end, 1):
        eps = np.prod(skip_coeff_xt[start:end])
        arr_eps[end-start-1, 0] = eps

        xzs = []
        for ii in range(start, end)[::-1]:
            base = float(skip_coeff_x0[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            xzs.append(base * factor)
        arr_xz[end-start-1, :end-start] = np.array(xzs)

        o2 = eps
        o1 = np.array(xzs).sum()

        g2 = np.sqrt(1 - alphas_bar[start * skip])
        g1 = np.sqrt(alphas_bar[start * skip])
        node_coeff[start, :] = np.array([g1, g2])

        print("start", start)
        print("pred", o1, o2)
        print("true", g1, g2)
    
    np.savez("ddim_%03d.npz"%num_step, past_x0_coeff=arr_xz, past_eps_coeff=arr_eps, node_coeff=node_coeff)
    print(arr_xz)
    print(arr_eps)

    return


def create_sde_coeff(skip_step=10):
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    std = np.sqrt(posterior_variance)

    coeff_x0 = np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar)
    coeff_xt = np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar)

    skip_alphas_bar = alphas_bar[::skip_step]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii - 1]

    skip_betas = 1 - skip_alphas

    skip_std = np.sqrt(skip_betas)

    skip_coeff_x0 = skip_betas*np.sqrt(skip_alphas_bar)/(1-skip_alphas_bar)
    skip_coeff_xt = (2 - np.sqrt(1-skip_betas) - skip_betas/(1-skip_alphas_bar))
    
    skip_coeff = skip_alphas, skip_alphas_bar, skip_std, skip_coeff_x0, skip_coeff_xt
    coeff = alphas, alphas_bar, std, coeff_x0, coeff_xt

    return skip_coeff, coeff


def sde_equivalent_coeff_tx():
    skip = 1
    skip_coeff, coeff = create_sde_coeff(skip)
    alphas, alphas_bar, std, coeff_x0, coeff_xt = coeff
    skip_alphas, skip_alphas_bar, skip_std, skip_coeff_x0, skip_coeff_xt = skip_coeff

    start, end = 1, 1000
    for start in range(0, 1000, 10):
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

    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    std = np.sqrt(posterior_variance)

    coeff_x0 = np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar)
    coeff_xt = np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar)

    skip_alphas_bar = alphas_bar[::skip_step]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii-1]

    skip_betas = 1 - skip_alphas

    skip_std = np.sqrt(skip_betas)

    skip_coeff_x0 = 0.5*skip_betas * np.sqrt(skip_alphas_bar)/(1-skip_alphas_bar)
    skip_coeff_xt = (2 - np.sqrt(1-skip_betas) - 0.5*skip_betas/(1-skip_alphas_bar))

    skip_coeff = skip_alphas, skip_alphas_bar, skip_std, skip_coeff_x0, skip_coeff_xt
    coeff = alphas, alphas_bar, std, coeff_x0, coeff_xt

    return skip_coeff, coeff


def ode_equivalent_coeff_tx():
    skip = 1
    skip_coeff, coeff = create_ddim_coeff(skip)
    alphas, alphas_bar, coeff_x0, coeff_xt = coeff
    skip_alphas, skip_alphas_bar, skip_coeff_x0, skip_coeff_xt = skip_coeff

    start, end = 0, 1000
    for start in range(0, 1000, 10):
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


def create_ddpm_coeff():
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    log_var = np.log(np.append(var[1], var[1:]))

    coeff_x0 = np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar)
    coeff_xt = np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar)
    coeff_xt2x0 = np.sqrt(1.0 / alphas_bar)
    coeff_eps2x0 = np.sqrt(1.0 / alphas_bar - 1)

    coeff_all = [alphas, alphas_bar, log_var, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0]

    return coeff_all


def skip_ddpm_coeff(coeff_all, skip_step=10):
    alphas, alphas_bar, log_var, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all

    roi_idxs = space_timesteps(1000, str(int(1000/skip_step)))
    roi_idxs = sorted(roi_idxs)

    skip_alphas_bar = alphas_bar[roi_idxs]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii - 1]

    skip_betas = 1 - skip_alphas
    skip_alphas_bar_prev = np.append(1.0, skip_alphas_bar[:-1])

    skip_var = skip_betas*(1.0-skip_alphas_bar_prev)/(1.0-skip_alphas_bar)
    skip_log_var = np.log(np.append(skip_var[1], skip_var[1:]))

    skip_coeff_x0 = np.sqrt(skip_alphas_bar_prev) * skip_betas / (1 - skip_alphas_bar)
    skip_coeff_xt = np.sqrt(skip_alphas) * (1 - skip_alphas_bar_prev) / (1 - skip_alphas_bar)

    skip_coeff_xt2x0 = np.sqrt(1.0 / skip_alphas_bar)
    skip_coeff_eps2x0 = np.sqrt(1.0 / skip_alphas_bar - 1)

    skip_coeff_all = [skip_alphas, skip_alphas_bar, skip_log_var,
                      skip_coeff_xt2x0, skip_coeff_eps2x0, skip_coeff_xt, skip_coeff_x0]

    return skip_coeff_all, roi_idxs


def ddpm_analyze_coeff_tx():
    skip = 55.555
    coeff_all = create_ddpm_coeff()
    skip_coeff_all, _ = skip_ddpm_coeff(coeff_all, skip)
    skip_alphas, skip_alphas_bar, skip_log_var, _, _, skip_coeff_xt, skip_coeff_x0 = skip_coeff_all
    skip_std = np.sqrt(np.exp(skip_log_var))
    
    num_step = int(1000 / skip)
    arr_eps, arr_xz = np.zeros([num_step, num_step + 1]), np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step, 2])

    end = num_step
    for start in range(0, end, 1):
        epss = [np.prod(skip_coeff_xt[start:end])]
        for ii in range(start, end)[::-1]:
            sigma = float(skip_std[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            epss.append(sigma * factor)
        arr_eps[end - start - 1, :1 + end - start] = np.array(epss)

        xzs = []
        for ii in range(start, end)[::-1]:
            base = float(skip_coeff_x0[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            xzs.append(base * factor)
        arr_xz[end - start - 1, :end - start] = np.array(xzs)

        o2 = np.linalg.norm(np.array(epss))
        o1 = np.array(xzs).sum()

        g2 = np.sqrt(1 - skip_alphas_bar[start])
        g1 = np.sqrt(skip_alphas_bar[start])
        node_coeff[start, :] = np.array([g1, g2])

        print("start", start)
        print("pred: %0.4f %0.4f"%(o1, o2))
        print("true: %0.4f %0.4f"%(g1, g2))
    
    names = ["s%02d" % ii for ii in range(num_step-1, -1, -1)]
    df = pd.DataFrame(arr_xz.round(3), columns=names, index=names)
    df["sum"] = arr_xz.sum(axis=1).round(3)
    df.to_csv("results/ddpm/ddpm_%03d.csv" % num_step)
    print(df)
    
    np.savez("results/ddpm/ddpm_%03d.npz" % num_step, past_x0_coeff=arr_xz, past_eps_coeff=arr_eps, node_coeff=node_coeff)
    print(arr_xz)
    print(arr_eps)
    return


def create_ddim_coeff():
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    coeff_xt2x0 = np.sqrt(1.0 / alphas_bar)
    coeff_eps2x0 = np.sqrt(1.0 / alphas_bar - 1)

    recified_factor = np.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar))

    coeff_x0 = np.sqrt(alphas_bar_prev) - recified_factor * np.sqrt(alphas_bar)
    coeff_xt = recified_factor

    coeff_all = alphas, alphas_bar, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0

    return coeff_all


def skip_ddim_coeff(coeff_all, skip_step=10):
    alphas, alphas_bar, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all

    roi_idxs = space_timesteps(1000, str(int(1000/skip_step)))
    roi_idxs = sorted(roi_idxs)

    skip_alphas_bar = alphas_bar[roi_idxs]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii - 1]

    skip_betas = 1 - skip_alphas
    skip_alphas_bar_prev = np.append(1.0, skip_alphas_bar[:-1])

    skip_coeff_xt2x0 = np.sqrt(1.0 / skip_alphas_bar)
    skip_coeff_eps2x0 = np.sqrt(1.0 / skip_alphas_bar - 1)

    skip_recified_factor = np.sqrt((1 - skip_alphas_bar_prev) / (1 - skip_alphas_bar))
    skip_coeff_x0 = np.sqrt(skip_alphas_bar_prev) - skip_recified_factor * np.sqrt(skip_alphas_bar)
    skip_coeff_xt = skip_recified_factor

    skip_coeff_all = skip_alphas, skip_alphas_bar, skip_coeff_xt2x0, skip_coeff_eps2x0, skip_coeff_xt, skip_coeff_x0

    return skip_coeff_all, roi_idxs


def ddim_analyze_coeff_tx():
    skip = 55.555
    coeff_all = create_ddim_coeff()
    skip_coeff_all, _ = skip_ddim_coeff(coeff_all, skip)
    skip_alphas, skip_alphas_bar, _, _, skip_coeff_xt, skip_coeff_x0 = skip_coeff_all

    num_step = int(1000/skip)
    arr_eps, arr_xz = np.zeros([num_step, num_step+1]), np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step, 2])

    end = num_step
    for start in range(0, end, 1):
        eps = np.prod(skip_coeff_xt[start:end])
        arr_eps[end-start-1, 0] = eps

        xzs = []
        for ii in range(start, end)[::-1]:
            base = float(skip_coeff_x0[ii])
            factor = float(np.prod(skip_coeff_xt[start:ii]))
            xzs.append(base * factor)
        arr_xz[end-start-1, :end-start] = np.array(xzs)

        o2 = eps
        o1 = np.array(xzs).sum()

        g2 = np.sqrt(1 - skip_alphas_bar[start])
        g1 = np.sqrt(skip_alphas_bar[start])
        node_coeff[start, :] = np.array([g1, g2])

        print("start", start)
        print("pred: %0.4f %0.4f"%(o1, o2))
        print("true: %0.4f %0.4f"%(g1, g2))
    
    names = ["s%02d"%ii for ii in range(num_step-1, -1, -1)]
    df = pd.DataFrame(arr_xz.round(3), columns=names, index=names)
    df["sum"] = arr_xz.sum(axis=1).round(3)
    df.to_csv("results/ddim/ddim_%03d.csv"%num_step)
    print(df)
    
    np.savez("results/ddim/ddim_%03d.npz" % num_step, past_x0_coeff=arr_xz, past_eps_coeff=arr_eps, node_coeff=node_coeff)
    print(arr_xz)
    print(arr_eps)
     
    return


def flow_analyze_coeff_tx():
    num_step = 50
    
    sigmas = np.linspace(0, 1, num_step+1)
    coeff_x0 = 1 - sigmas[:-1]/sigmas[1:]
    coeff_xt = sigmas[:-1]/sigmas[1:]
     
    arr_eps, arr_xz = np.zeros([num_step, num_step+1]), np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step, 2])

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
        node_coeff[start, :] = np.array([g1, g2])

        print("start", start)
        print("pred: %0.4f %0.4f"%(o1, o2))
        print("true: %0.4f %0.4f"%(g1, g2))

    names = ["s%02d" % ii for ii in range(num_step-1, -1, -1)]
    df = pd.DataFrame(arr_xz.round(3), columns=names, index=names)
    df["sum"] = arr_xz.sum(axis=1).round(3)
    df.to_csv("results/flow_euler/flow_euler_%03d.csv" % num_step)
    print(df)

    np.savez("results/flow_euler/flow_euler_%03d.npz" % num_step, past_x0_coeff=arr_xz, past_eps_coeff=arr_eps, node_coeff=node_coeff)
    print(arr_xz)
    print(arr_eps)

    return


def convert_npz_to_csv():
    work_dir = "D:\codes\\WeSee\\NaturalDiffusion\\results\\natural_inference"
    npz_name = "step_15_weight_173.npz"
    
    past_x0_coeff, past_eps_coeff, node_coeff = np.load(os.path.join(work_dir, npz_name)).values()
    x0_coeff = past_x0_coeff / past_x0_coeff.diagonal()[:, None]
    columns = ["t=%0.3f" % val for val in node_coeff[1:, 0]]
    df = pd.DataFrame(x0_coeff.round(3), columns=columns, index=columns)
    df["image_amplitude"] = node_coeff[1:, 1].round(3)
    df.to_csv(os.path.join(work_dir, npz_name[:-4]+".csv"))
    
    return


if __name__ == "__main__":
    flow_analyze_coeff_tx()
    # ddpm_analyze_coeff_tx()
    # ddim_analyze_coeff_tx()
    # ddpm_equivalent_coeff_tx()
    # ddim_equivalent_coeff_tx()