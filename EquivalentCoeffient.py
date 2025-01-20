
import numpy as np


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


def create_ddpm_coeff(skip_step=10):
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
    skip = 1
    skip_coeff, coeff = create_ddpm_coeff(skip)
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

        g2 = np.sqrt(1 - alphas_bar[start*skip])
        g1 = np.sqrt(alphas_bar[start*skip])

        print("start", start)
        print("pred", o1, o2)
        print("true", g1, g2)
        
    return


def create_ddim_coeff(skip_step):
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
    skip = 20
    skip_coeff, coeff = create_ddim_coeff(skip)
    alphas, alphas_bar, coeff_x0, coeff_xt = coeff
    skip_alphas, skip_alphas_bar, skip_coeff_x0, skip_coeff_xt = skip_coeff

    start, end = 5, 50
    for start in range(0, 50, 5):
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
    
    # coeff_x0 = torch.from_numpy(coeff_x0).to(dtype=torch.float32, device=device)
    # coeff_xt = torch.from_numpy(coeff_xt).to(dtype=torch.float32, device=device)
    # coeff_xt2x0 = torch.from_numpy(coeff_xt2x0).to(dtype=torch.float32, device=device)
    # coeff_eps2x0 = torch.from_numpy(coeff_eps2x0).to(dtype=torch.float32, device=device)
    # log_var = torch.from_numpy(posterior_log_variance_clipped).to(dtype=torch.float32, device=device)
    # sigma = torch.from_numpy().to(dtype=torch.float32, device=device)

    coeff_all = [alphas, alphas_bar, log_var, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0]
    
    return coeff_all


def skip_ddpm_coeff(coeff_all, skip_step=10):
    alphas, alphas_bar, log_var, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all
    
    if skip_step == 4000:
        roi_idxs = key_timesteps
    else:
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

def skip_ddim_coeff(coeff_all, skip_step=10):
    alphas, alphas_bar, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all


    if skip_step == 4000:
        roi_idxs = key_timesteps
    else:
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