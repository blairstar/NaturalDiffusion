import numpy as np

import th_deis as tdeis
from th_deis.sampler import get_sampler_t_ab, get_sampler_rho_ab
from th_deis.sde import get_rev_ts
from th_deis.multistep import get_ab_eps_coef
import jax.numpy as jnp
from th_deis.helper import jax2th, th2jax
import sympy
from sympy import symbols
import functools
np.set_printoptions(suppress=True, linewidth=300, precision=5)
from collections import OrderedDict

expr_pool = OrderedDict()


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



def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def ab_step(x, ei_coef, new_eps, eps_pred):
    x_coef, eps_coef = ei_coef[0], ei_coef[1:]
    full_eps_pred = [new_eps, *eps_pred]
    rtn = x_coef * x
    for cur_coef, cur_eps in zip(eps_coef, full_eps_pred):
        rtn += cur_coef * cur_eps
    return rtn, full_eps_pred[:-1]


def get_sampler_t_ab(sde, eps_fn, ts_phase, ts_order, num_step, ab_order):
    jax_rev_ts = get_rev_ts(sde, num_step, ts_order, ts_phase=ts_phase)

    x_coef = sde.psi(jax_rev_ts[:-1], jax_rev_ts[1:])
    eps_coef = get_ab_eps_coef(sde, ab_order, jax_rev_ts, ab_order)
    jax_ab_coef = jnp.concatenate([x_coef[:, None], eps_coef], axis=1)
    # th_rev_ts, th_ab_coef = jax2th(jax_rev_ts), jax2th(jax_ab_coef)
    rev_ts, ab_coef = np.array(jax_rev_ts), np.array(jax_ab_coef)
     
    def sampler(xT):
        # rev_ts, ab_coef = th_rev_ts.to(xT.device), th_ab_coef.to(xT.device)

        def ab_body_fn(i, val):
            x, eps_pred = val
            s_t = rev_ts[i]

            new_eps = eps_fn(x, s_t)
            new_x, new_eps_pred = ab_step(x, ab_coef[i], new_eps, eps_pred)

            add_item("x_%0.4f"%rev_ts[i+1], new_x)
            return new_x, new_eps_pred

        eps_pred = [xT, ] * ab_order
        img, _ = fori_loop(0, num_step, ab_body_fn, (xT, eps_pred))
        return img

    return sampler, rev_ts


def calc_x_eps_coeff(beta_min, beta_max, t):
    log_x_coeff = -0.25*t**2 * (beta_max - beta_min) - 0.5*t*beta_min
    x_coeff = np.exp(log_x_coeff)
    eps_coeff = jnp.sqrt(1 - np.exp(2. * log_x_coeff))
    return x_coeff, eps_coeff


def eps_fn(x_t, t, beta_min, beta_max):
    y_coeff, eps_coeff = calc_x_eps_coeff(beta_min, beta_max, t)

    y_t = symbols("y_%0.4f" % t)
    add_item("y_%0.4f" % t, y_t)

    pred_eps = (x_t - y_coeff * y_t) / eps_coeff
    return pred_eps


def sampling_tab_tx():
    num_step = 10
    sampling_eps, T = 0.001, 1
    beta_min, beta_max = 0.1, 20
    t2alpha_fn, alpha2t_fn = tdeis.get_linear_alpha_fns(beta_min, beta_max)
    vpsde = tdeis.VPSDE(t2alpha_fn, alpha2t_fn, sampling_eps, T)
    
    eps_fn_partial = functools.partial(eps_fn, beta_min=beta_min, beta_max=beta_max)
    t_ab_fn, rev_ts = get_sampler_t_ab(vpsde, eps_fn_partial, "t", 2, num_step, ab_order=3)

    eps_t = symbols("eps_%0.4f" % rev_ts[0])
    add_item("eps_%0.4f" % rev_ts[0], eps_t)
    add_item("x_%0.4f" % rev_ts[0], eps_t*1.0)
    
    out = t_ab_fn(eps_t)

    ys = get_y_symbols()
    epss = get_eps_symbols()
    
    past_xstart_coeff = np.zeros([num_step, num_step])
    past_epsilon_coeff = np.zeros([num_step, num_step])
    node_coeff = np.zeros([num_step+1, 3])
    
    for kk, t in enumerate(rev_ts):
        x_t = get_item("x_%0.4f" % t)
        
        true_y_coeff, true_eps_coeff = calc_x_eps_coeff(beta_min, beta_max, t)

        print("t", t)
        y_coeffs = show_symbol_coeff(x_t, ys)
        print("y result", np.sum(y_coeffs), true_y_coeff)

        eps_coeffs = show_symbol_coeff(x_t, epss)
        print("eps result", np.linalg.norm(eps_coeffs), true_eps_coeff)
        print("")

        node_coeff[kk, :] = np.array([t, true_y_coeff, true_eps_coeff])
        if not np.isclose(t, 1.0):
            past_xstart_coeff[kk-1, :len(y_coeffs)] = np.array(y_coeffs)
            past_epsilon_coeff[kk-1, :len(eps_coeffs)] = np.array(eps_coeffs)
    
    print(past_xstart_coeff)
    print(past_epsilon_coeff)
    print(node_coeff)
    
    np.savez("tab_%s.npz"%num_step, past_xstart_coeff=past_xstart_coeff,
             past_epsilon_coeff=past_epsilon_coeff, node_coeff=node_coeff)
    
    return


if __name__ == "__main__":
    sampling_tab_tx()
