import numpy as np


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
    