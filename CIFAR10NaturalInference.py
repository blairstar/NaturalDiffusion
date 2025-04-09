
# # We use the pre-trained model released by ScoreSDE. Please download vp/cifar10_ddpm_continuous/checkpoint_8.pth.

import os, sys
sys.path.append("deps")
sys.path.append("deps/score_sde_pytorch")

from models import ncsnpp
from models import ddpm as ddpm_model
from sde_lib import VPSDE

from models import utils as mutils
import datasets
from configs.vp import cifar10_ddpmpp_continuous as configs
import matplotlib.pyplot as plt

import th_deis as tdeis
import jax.random as random
import cv2
from PIL import Image
import pandas as pd

import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint
import numpy as np

from pytorch_fid.inception import InceptionV3
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.fid_score import calculate_frechet_distance
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from dpm_solver_pytorch import NoiseScheduleVP, DPM_Solver

import itertools
    

np.set_printoptions(suppress=True, linewidth=600, precision=2)
torch.set_printoptions(sci_mode=False, linewidth=600, precision=2)


def get_activation(imgs, model, dims, device):
    model.eval()
    
    batch_size = 50
    pred_arr = np.empty((len(imgs), dims))

    start_idx = 0
    for ii in tqdm(range(0, len(imgs), batch_size)):
        batch = imgs[ii:ii+batch_size]
        batch = batch.to(dtype=torch.float32, device=device)/255
        batch = batch.permute(0, 3, 1, 2)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx: start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calc_fid(imgs, ref_path, device):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    act = get_activation(imgs, model, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    
    file_obj = np.load(ref_path)
    ref_mu, ref_sigma = file_obj['mu'], file_obj['sigma']
    
    fid_value = calculate_frechet_distance(ref_mu, ref_sigma, mu, sigma)
    return fid_value


def calc_fid_tx():
    import glob
    
    imgs = []
    for path in sorted(glob.glob("samples/tab_t2_ab3_s10_cmp/*.png"))[:]:
        img = Image.open(path).convert('RGB')
        imgs.append(torch.from_numpy(np.array(img)))
    imgs = torch.stack(imgs)
    
    fid_value = calc_fid(imgs, "./cifar10_mu_sigma.npz", "cuda")
    print(fid_value)
    return


def image_grid(x, config):
    size = config.data.image_size
    channels = config.data.num_channels
    img = x.reshape(-1, size, size, channels)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return img


def show_samples(x, path, config):
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = image_grid(x, config)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    return


@torch.no_grad()
def deis_sampling_tx():
    # # to do: set checkpoint path and batch_size
    batch_size = 500
    ckpt_filename = os.path.expanduser("deps/score_sde_pytorch/checkpoint_8.pth")
    assert os.path.exists(ckpt_filename)
    
    config = configs.get_config()
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3

    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size

    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())

    t2alpha_fn, alpha2t_fn = tdeis.get_linear_alpha_fns(sde.beta_0, sde.beta_1)
    vpsde = tdeis.VPSDE(t2alpha_fn, alpha2t_fn, sampling_eps, sde.T)

    score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=True)

    def eps_fn(x, scalar_t):
        vec_t = scalar_t * torch.ones(x.shape[0], device=config.device)
        score = score_fn(x, vec_t)
        std = sde.marginal_prob(torch.zeros_like(score), vec_t)[1]
        eps = -1 * score * std[:, None, None, None]
        return eps

    num_step = 5

    torch.manual_seed(888)
    
    sample_count = 50*1000
    bz = batch_size
    num = int(np.ceil(sample_count/bz))
    
    ts_phases = ["t", "rho"]
    method = ["t_ab", "rho_ab", "rho_rk"]
    ab_orders = [2, 3]

    infos = []
    for param in itertools.product(ts_phases, method, ab_orders):
        ts_phase, method, ab_order = param

        name = f"{num_step}__{ts_phase}__{method}__{ab_order}"
        print(name)
        
        sampling_func = tdeis.get_sampler(vpsde, eps_fn, ts_phase, 2, num_step, method=method, ab_order=ab_order)
        
        all_batch = []
        for ii in range(num):
            if ii%25 == 0:
                print("processing", ii)
            noise = torch.randn(bz, 3, 32, 32, dtype=torch.float32, device=config.device)
            out = inverse_scaler(sampling_func(noise))
            all_batch.append(to_pixel(out))
            
        all_batch = torch.concatenate(all_batch)
        fid_value = calc_fid(all_batch, "./cifar10_mu_sigma.npz", config.device)
        
        infos.append([num_step, ts_phase, method, ab_order, fid_value])
        
        print(name, fid_value)
        print("")
    
    df = pd.DataFrame(infos, columns=["num_step", "ts_phase", "method", "ab_order", "fid_value"])
    print(df.sort_values(by="fid_value"))
    
    return


def save_output(out, batch_idx, dir_path):
    out = out.permute(0, 2, 3, 1).detach().cpu().numpy()
    out = np.clip(out*255, 0, 255).astype(np.uint8)
    for jj in range(out.shape[0]):
        img = out[jj]
        path = os.path.join(dir_path, "%04d_%03d.png"%(batch_idx, jj))
        cv2.imwrite(path, img[:, :, ::-1])
    return


def to_pixel(batch):
    batch = batch.permute(0, 2, 3, 1).detach().cpu().numpy()
    batch = np.clip(batch*255, 0, 255).astype(np.uint8)
    batch = torch.from_numpy(batch)
    return batch


@torch.no_grad()
def data_fn(score_fn, xt, t, x_coeff, eps_coeff, device):
    vec_t = t * torch.ones(xt.shape[0], device=device)
    score = score_fn(xt, vec_t)

    xt = xt.to(dtype=torch.float64)
    score = score.to(dtype=torch.float64)
    eps_coeff = torch.tensor(eps_coeff, dtype=torch.float64, device=device)
    x_coeff = torch.tensor(x_coeff, dtype=torch.float64, device=device)

    pred_data = (score * eps_coeff**2 + xt)/x_coeff
    return pred_data


def weighted_sum(past_x0_coeff, seq_x0):
    out = torch.zeros_like(seq_x0[0])
    for ii, x0 in enumerate(seq_x0):
        out += x0 * past_x0_coeff[ii]
    out = out.to(dtype=torch.float32)
    return out


@torch.no_grad()
def natural_inference_tx():
    # # to do: set checkpoint path, batch_size and weight_path
    batch_size = 500
    ckpt_filename = os.path.expanduser("deps/score_sde_pytorch/checkpoint_8.pth")
    weight_path = "weights/step_5_weight_00.npz"
    assert os.path.exists(ckpt_filename)
    
    config = configs.get_config()
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3

    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size

    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())

    t2alpha_fn, alpha2t_fn = tdeis.get_linear_alpha_fns(sde.beta_0, sde.beta_1)
    vpsde = tdeis.VPSDE(t2alpha_fn, alpha2t_fn, sampling_eps, sde.T)

    score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=True)
    
    # # Be careful! Make sure that the past_x0_coeff have been normalized to the marginal signal coefficients
    past_x0_coeff, past_eps_coeff, node_coeff = np.load(weight_path).values()
   
    print(past_x0_coeff/np.diag(past_x0_coeff)[:, None])
    print(weight_path)
    
    ts = node_coeff[:, 0]
    num_step = ts.shape[0] - 1
    
    sample_count = 50 * 1000
    bz = batch_size
    num = int(np.ceil(sample_count / bz))

    torch.manual_seed(888)
    
    all_batch = []
    for ii in range(num):
        print("processing", ii)
        noise = torch.randn(bz, 3, 32, 32, dtype=torch.float32, device=config.device)
        
        seq_x0 = []
        next_model_input = noise
        for kk in range(num_step):
            t = ts[kk]
            
            model_input = next_model_input
            pred_x0 = data_fn(score_fn, model_input, t, node_coeff[kk, 1], node_coeff[kk, 2], config.device)
            seq_x0.append(pred_x0)
            
            next_x0 = weighted_sum(past_x0_coeff[kk], seq_x0)
            # next_eps = node_coeff[kk+1, 2]*noise
            next_eps = past_eps_coeff[kk, 0]*noise
            next_model_input = next_x0 + next_eps
        out = next_model_input
        
        out = inverse_scaler(out)
        all_batch.append(to_pixel(out))
        
    all_batch = torch.concatenate(all_batch)
    fid_value = calc_fid(all_batch, "./cifar10_mu_sigma.npz", config.device)
    print(fid_value)
    print(weight_path)
    print(past_x0_coeff/np.diag(past_x0_coeff)[:, None])
    
    return


def get_noise_fn(model_fn):
    def noise_fn(x, t):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        noise = model_fn(x, labels)
        return noise
    return noise_fn


@torch.no_grad()
def dpm_solver_tx():
    # # to do: set checkpoint path and batch_size
    batch_size = 500
    ckpt_filename = os.path.expanduser("deps/score_sde_pytorch/checkpoint_8.pth")
    assert os.path.exists(ckpt_filename)
    
    config = configs.get_config()
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3

    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size

    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())
    
    num_step = 5

    model_fn = mutils.get_model_fn(score_model, train=False)
    noise_fn = get_noise_fn(model_fn)
    
    print("beta", sde.beta_0, sde.beta_1) 
    
    algorithm_types = ["dpmsolver", "dpmsolver++"]
    methods = ["singlestep", "multistep"]
    skip_types = ["time_quadratic"]
    correcting_x0_fns = [None, "dynamic_thresholding"]
    orders = [2, 3]
    
    infos = []
    for param in itertools.product(algorithm_types, methods, skip_types, correcting_x0_fns, orders):
        algorithm_type, method, skip_type, correcting_x0_fn, order = param
        ns = NoiseScheduleVP('linear', continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)
        
        dpm_solver = DPM_Solver(noise_fn, ns, algorithm_type=algorithm_type, correcting_x0_fn=correcting_x0_fn)

        print("num step: %d" % num_step)
        name = f"{num_step}_{algorithm_type}_{method}_{skip_type}_{correcting_x0_fn}_{order}"
        print(name)

        sample_count = 50 * 1000
        bz = batch_size
        num = int(np.ceil(sample_count / bz))

        torch.manual_seed(888)

        all_batch = []
        for ii in range(num):
            if ii%20 == 0:
                print("processing", ii)
            noise = torch.randn(bz, 3, 32, 32, dtype=torch.float32, device=config.device)

            out = dpm_solver.sample(noise, steps=num_step, t_start=sde.T, t_end=sampling_eps, order=order,
                                    skip_type=skip_type, method=method, denoise_to_zero=False, lower_order_final=False)

            out = inverse_scaler(out)
            all_batch.append(to_pixel(out))

        all_batch = torch.concatenate(all_batch)
        fid_value = calc_fid(all_batch, "./cifar10_mu_sigma.npz", config.device)
        infos.append([num_step, algorithm_type, method, skip_type, correcting_x0_fn, order, fid_value])
        print(num_step, algorithm_type, method, skip_type, correcting_x0_fn, order, fid_value)
        print("")

    columns = ["num_step", "alg_type", "mult", "time", "thresh", "order", "fid"]
    df = pd.DataFrame(infos, columns=columns)
    
    print(df[df["alg_type"] == "dpmsolver"].sort_values("fid"))
    print(df[df["alg_type"] == "dpmsolver++"].sort_values("fid"))

    return



if __name__ == "__main__":
    # deis_sampling_tx()
    natural_inference_tx()
    # calc_fid_tx()
    # dpm_solver_tx()