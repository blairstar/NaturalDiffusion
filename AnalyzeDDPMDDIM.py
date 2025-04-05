
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scienceplots
import sympy
from sympy import symbols


from Utils import draw_marginal_coeff, CAnalyzer


np.set_printoptions(suppress=True, linewidth=200, precision=4)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)



# # copied from https://github.com/openai/improved-diffusion
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


def create_ddpm_coeff():
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_prev = np.append(1.0, alphas_bar[:-1])

    var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    log_var = np.log(np.append(1E-5, var[1:]))

    coeff_x0 = np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar)
    coeff_xt = np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar)
    coeff_xt2x0 = np.sqrt(1.0 / alphas_bar)
    coeff_eps2x0 = np.sqrt(1.0 / alphas_bar - 1)

    coeff_all = [alphas, alphas_bar, log_var, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0]

    return coeff_all


def skip_ddpm_coeff(coeff_all, num_step=20):
    alphas, alphas_bar, log_var, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all

    roi_idxs = space_timesteps(1000, str(int(num_step)))
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


def ddpm_analyze_coeff(num_step=20):
    coeff_all = create_ddpm_coeff()
    skip_coeff_all, skip_idxs = skip_ddpm_coeff(coeff_all, num_step)
    skip_alphas, skip_alphas_bar, skip_log_var, _, _, skip_coeff_xt, skip_coeff_x0 = skip_coeff_all
    skip_std = np.sqrt(np.exp(skip_log_var))
    
    arr_eps, arr_xz = np.zeros([num_step, num_step + 1]), np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step, 3])

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
        
        if start == 0:
            time_idx, g2, g1 = -1, 0.0, 1.0
        else:
            time_idx = skip_idxs[start-1]
            g2 = np.sqrt(1 - skip_alphas_bar[start-1])
            g1 = np.sqrt(skip_alphas_bar[start-1])
        
        node_coeff[end-start-1, :] = np.array([time_idx, g1, g2])

        print("start", start, "time_idx", time_idx)
        print("pred: %0.4f %0.4f"%(o1, o2))
        print("true: %0.4f %0.4f"%(g1, g2))

    node_coeff = np.vstack([np.array([999, 0.0, 1.0]), node_coeff])
    
    names = ["%03d"%node_coeff[ii, 0] for ii in range(0, num_step+1)]
    df = pd.DataFrame(arr_xz.round(3), columns=names[:-1], index=names[1:])
    df["sum"] = arr_xz.sum(axis=1).round(3)
    df.to_csv("results/ddpm/ddpm_%03d.csv" % num_step)
    print(df)

    draw_marginal_coeff(arr_xz, arr_eps, node_coeff, "results/ddpm/ddpm_%03d.jpg"%num_step)

    np.savez("results/ddpm/ddpm_%03d.npz" % num_step, past_x0_coeff=arr_xz, past_eps_coeff=arr_eps, node_coeff=node_coeff)
    print(arr_xz)
    print(arr_eps)
    print(node_coeff)
    return


def ddpm_simpy_analyze_coeff(num_step=20):
    coeff_all = create_ddpm_coeff()
    skip_coeff_all, skip_idxs = skip_ddpm_coeff(coeff_all, num_step)
    skip_alphas, skip_alphas_bar, skip_log_var, skip_coeff_xt2x0, skip_coeff_eps2x0, skip_coeff_xt, skip_coeff_x0 = skip_coeff_all
    skip_stds = np.sqrt(np.exp(skip_log_var))

    analyzer = CAnalyzer()

    # reverse sequence
    # when step_idx = -1, skip_alphas_bar = 1, this means that denoise to zero.
    step_idxs = skip_idxs[::-1] + [-1]
    alphas_bar = np.append(skip_alphas_bar[::-1], 1)
    coeff_xt = np.append(skip_coeff_xt[::-1], np.nan)
    coeff_x0 = np.append(skip_coeff_x0[::-1], np.nan)
    stds = np.append(skip_stds[::-1], 0)

    t = step_idxs[0]
    eps_t = symbols("eps_%03d"%t)
    analyzer.add_item("eps_%03d"%t, eps_t)
    analyzer.add_item("x_%03d"%t, eps_t)
    
    for ii in range(0, num_step):
        s, t = step_idxs[ii], step_idxs[ii+1]
        x_s = analyzer.get_item("x_%03d"%s)
        
        y_s = symbols("y_%03d"%s)            # y_s is predicted x_0 in the s_th step
        analyzer.add_item("y_%03d"%s, y_s)
        
        # calculate posterior's mean
        mean_t = coeff_xt[ii] * x_s + coeff_x0[ii] * y_s
        
        eps_t = symbols("eps_%03d"%t)
        analyzer.add_item("eps_%03d"%t, eps_t)
        
        # sample x_t from posterior's mean
        x_t = mean_t + stds[ii]*eps_t
        
        analyzer.add_item("x_%03d"%t, x_t)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()

    past_xstart_coeff = np.zeros([num_step, num_step])
    past_epsilon_coeff = np.zeros([num_step, num_step+1])
    node_coeff = np.zeros([num_step+1, 3])

    for kk, t in enumerate(step_idxs):
        x_t = analyzer.get_item("x_%03d"%t)
        
        true_y_coeff = np.sqrt(alphas_bar[kk])
        true_eps_coeff = np.sqrt(1 - true_y_coeff**2)

        print("t", t)
        y_coeffs = analyzer.show_symbol_coeff(x_t, ys)
        print("y result", np.sum(y_coeffs), true_y_coeff)

        eps_coeffs = analyzer.show_symbol_coeff(x_t, epss)
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_coeff)
        print("")

        node_coeff[kk, :] = np.array([t, true_y_coeff, true_eps_coeff])
        if kk > 0:
            past_xstart_coeff[kk-1, :len(y_coeffs)] = np.array(y_coeffs)
            past_epsilon_coeff[kk-1, :len(eps_coeffs)] = np.array(eps_coeffs)

    names = ["%03d" % node_coeff[ii, 0] for ii in range(0, num_step+1)]
    df = pd.DataFrame(past_xstart_coeff.round(3), columns=names[:-1], index=names[1:])
    df["sum"] = past_xstart_coeff.sum(axis=1).round(3)
    df.to_csv("results/ddpm/ddpm_simpy_%03d.csv" % num_step)
    print(df)

    draw_marginal_coeff(past_xstart_coeff, past_epsilon_coeff,
                        node_coeff, "results/ddpm/ddpm_simpy_%03d.jpg" % num_step)

    np.savez("results/ddpm/ddpm_simpy_%03d.npz"%num_step, past_xstart_coeff=past_xstart_coeff,
             past_epsilon_coeff=past_epsilon_coeff, node_coeff=node_coeff)
    
    print(past_xstart_coeff)
    print(past_epsilon_coeff)
    print(node_coeff)

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


def skip_ddim_coeff(coeff_all, num_step=10):
    alphas, alphas_bar, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all

    roi_idxs = space_timesteps(1000, str(int(num_step)))
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


def ddim_analyze_coeff(num_step=20):
    coeff_all = create_ddim_coeff()
    skip_coeff_all, skip_idxs = skip_ddim_coeff(coeff_all, num_step)
    skip_alphas, skip_alphas_bar, _, _, skip_coeff_xt, skip_coeff_x0 = skip_coeff_all

    arr_eps, arr_xz = np.zeros([num_step, num_step+1]), np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step, 3])

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

        if start == 0:
            time_idx, g2, g1 = -1, 0.0, 1.0
        else:
            time_idx = skip_idxs[start-1]
            g2 = np.sqrt(1 - skip_alphas_bar[start-1])
            g1 = np.sqrt(skip_alphas_bar[start-1])
        node_coeff[end-start-1, :] = np.array([time_idx, g1, g2])

        print("start", start, "time_idx", time_idx)
        print("pred: %0.4f %0.4f"%(o1, o2))
        print("true: %0.4f %0.4f"%(g1, g2))

    node_coeff = np.vstack([np.array([999, 0.0, 1.0]), node_coeff])
    
    names = ["%03d"%node_coeff[ii, 0] for ii in range(0, num_step+1)]
    df = pd.DataFrame(arr_xz.round(3), columns=names[:-1], index=names[1:])
    df["sum"] = arr_xz.sum(axis=1).round(3)
    df.to_csv("results/ddim/ddim_%03d.csv"%num_step)
    print(df)

    draw_marginal_coeff(arr_xz, arr_eps, node_coeff, "results/ddim/ddim_%03d.jpg"%num_step)
    
    np.savez("results/ddim/ddim_%03d.npz" % num_step, past_x0_coeff=arr_xz, past_eps_coeff=arr_eps, node_coeff=node_coeff)
    print(arr_xz)
    print(arr_eps)
    print(node_coeff)

    return


def ddim_simpy_analyze_coeff(num_step=20):
    coeff_all = create_ddim_coeff()
    skip_coeff_all, skip_idxs = skip_ddim_coeff(coeff_all, num_step)
    skip_alphas, skip_alphas_bar, skip_coeff_xt2x0, skip_coeff_eps2x0, skip_coeff_xt, skip_coeff_x0 = skip_coeff_all

    analyzer = CAnalyzer()
    
    # reverse sequence
    # when step_idx = -1, skip_alphas_bar = 1, this means that denoise to zero.
    step_idxs = skip_idxs[::-1] + [-1]
    alphas_bar = np.append(skip_alphas_bar[::-1], 1)
    coeff_xt = np.append(skip_coeff_xt[::-1], np.nan)
    coeff_x0 = np.append(skip_coeff_x0[::-1], np.nan)

    t = step_idxs[0]
    eps_t = symbols("eps_%03d" % t)
    analyzer.add_item("eps_%03d" % t, eps_t)
    analyzer.add_item("x_%03d" % t, eps_t)

    for ii in range(0, num_step):
        s, t = step_idxs[ii], step_idxs[ii+1]
        x_s = analyzer.get_item("x_%03d" % s)

        y_s = symbols("y_%03d" % s)  # y_s is predicted x_0 in the s_th step
        analyzer.add_item("y_%03d" % s, y_s)

        # calculate x_t from predicted x_0 and x_{t-1}
        x_t = coeff_xt[ii] * x_s + coeff_x0[ii] * y_s

        analyzer.add_item("x_%03d" % t, x_t)

    ys = analyzer.get_y_symbols()
    epss = analyzer.get_eps_symbols()

    past_xstart_coeff = np.zeros([num_step, num_step])
    past_epsilon_coeff = np.zeros([num_step, num_step + 1])
    node_coeff = np.zeros([num_step + 1, 3])

    for kk, t in enumerate(step_idxs):
        x_t = analyzer.get_item("x_%03d" % t)

        true_y_coeff = np.sqrt(alphas_bar[kk])
        true_eps_coeff = np.sqrt(1 - true_y_coeff ** 2)

        print("t", t)
        y_coeffs = analyzer.show_symbol_coeff(x_t, ys)
        print("y result", np.sum(y_coeffs), true_y_coeff)

        eps_coeffs = analyzer.show_symbol_coeff(x_t, epss)
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_coeff)
        print("")

        node_coeff[kk, :] = np.array([t, true_y_coeff, true_eps_coeff])
        if kk > 0:
            past_xstart_coeff[kk - 1, :len(y_coeffs)] = np.array(y_coeffs)
            past_epsilon_coeff[kk - 1, :len(eps_coeffs)] = np.array(eps_coeffs)

    names = ["%03d" % node_coeff[ii, 0] for ii in range(0, num_step+1)]
    df = pd.DataFrame(past_xstart_coeff.round(3), columns=names[:-1], index=names[1:])
    df["sum"] = past_xstart_coeff.sum(axis=1).round(3)
    df.to_csv("results/ddim/ddim_simpy_%03d.csv" % num_step)
    print(df)

    draw_marginal_coeff(past_xstart_coeff, past_epsilon_coeff,
                        node_coeff, "results/ddim/ddim_simpy_%03d.jpg" % num_step)

    np.savez("results/ddim/ddim_simpy_%03d.npz" % num_step, past_xstart_coeff=past_xstart_coeff,
             past_epsilon_coeff=past_epsilon_coeff, node_coeff=node_coeff)

    print(past_xstart_coeff)
    print(past_epsilon_coeff)
    print(node_coeff)
    return


def ddpm_analyze_coeff_tx():
    for step in [18, 24, 100, 500]:
        ddpm_analyze_coeff(step)
    return


def ddim_analyze_coeff_tx():
    for step in [18, 24, 100, 500]:
        ddim_analyze_coeff(step)
    return


def ddpm_simpy_analyze_coeff_tx():
    for step in [18, 24, 100, 200]:
        ddpm_simpy_analyze_coeff(step)
    return


def ddim_simpy_analyze_coeff_tx():
    for step in [18, 24, 100, 200]:
        ddim_simpy_analyze_coeff(step)
    return


def convert_npz_to_csv():
    work_dir = "results\\natural_inference"
    npz_name = "step_15_weight_173.npz"
    
    past_x0_coeff, past_eps_coeff, node_coeff = np.load(os.path.join(work_dir, npz_name)).values()
    x0_coeff = past_x0_coeff / past_x0_coeff.diagonal()[:, None]
    columns = ["t=%0.3f" % val for val in node_coeff[1:, 0]]
    df = pd.DataFrame(x0_coeff.round(3), columns=columns, index=columns)
    df["image_amplitude"] = node_coeff[1:, 1].round(3)
    df.to_csv(os.path.join(work_dir, npz_name[:-4]+".csv"))
    
    return


if __name__ == "__main__":
    '''
    Here, we offer two options: one is to compute directly with the analytical expression, and the other is to leverage SimPy for automatic computation.
    For SimPy, when the number of steps exceeds 200, the computation becomes relatively slow.
    '''
    # ddpm_analyze_coeff_tx()
    # ddim_analyze_coeff_tx()
    ddpm_simpy_analyze_coeff_tx()
    ddim_simpy_analyze_coeff_tx()