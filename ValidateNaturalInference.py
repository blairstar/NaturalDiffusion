
import os, sys
sys.path.append("deps")
from diffusers.models import AutoencoderKL
from models import DiT_models
import torch
import numpy as np
import copy
from torchvision.utils import save_image



torch.set_printoptions(sci_mode=False, precision=6, linewidth=200)
np.set_printoptions(suppress=True, precision=6, linewidth=200)


def make_path(path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


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


def skip_ddpm_coeff(coeff_all, num_step=50):
    alphas, alphas_bar, log_var, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all

    roi_idxs = space_timesteps(1000, str(num_step))
    roi_idxs = sorted(roi_idxs)

    skip_alphas_bar = alphas_bar[roi_idxs]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii - 1]

    skip_betas = 1 - skip_alphas
    skip_alphas_bar_prev = np.append(1.0, skip_alphas_bar[:-1])

    skip_var = skip_betas * (1.0 - skip_alphas_bar_prev) / (1.0 - skip_alphas_bar)
    skip_log_var = np.log(np.append(1E-5, skip_var[1:]))

    skip_coeff_x0 = np.sqrt(skip_alphas_bar_prev) * skip_betas / (1 - skip_alphas_bar)
    skip_coeff_xt = np.sqrt(skip_alphas) * (1 - skip_alphas_bar_prev) / (1 - skip_alphas_bar)

    skip_coeff_xt2x0 = np.sqrt(1.0 / skip_alphas_bar)
    skip_coeff_eps2x0 = np.sqrt(1.0 / skip_alphas_bar - 1)

    skip_coeff_all = [skip_alphas, skip_alphas_bar, skip_log_var,
                      skip_coeff_xt2x0, skip_coeff_eps2x0, skip_coeff_xt, skip_coeff_x0]

    return skip_coeff_all, roi_idxs


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


def skip_ddim_coeff(coeff_all, num_step=50):
    alphas, alphas_bar, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all

    roi_idxs = space_timesteps(1000, str(num_step))
    roi_idxs = sorted(roi_idxs)

    skip_alphas_bar = alphas_bar[roi_idxs]

    skip_alphas = np.zeros_like(skip_alphas_bar)
    skip_alphas[0] = skip_alphas_bar[0]
    for ii in range(1, len(skip_alphas_bar)):
        skip_alphas[ii] = skip_alphas_bar[ii] / skip_alphas_bar[ii - 1]

    skip_alphas_bar_prev = np.append(1.0, skip_alphas_bar[:-1])

    skip_coeff_xt2x0 = np.sqrt(1.0 / skip_alphas_bar)
    skip_coeff_eps2x0 = np.sqrt(1.0 / skip_alphas_bar - 1)

    skip_recified_factor = np.sqrt((1 - skip_alphas_bar_prev) / (1 - skip_alphas_bar))
    skip_coeff_x0 = np.sqrt(skip_alphas_bar_prev) - skip_recified_factor * np.sqrt(skip_alphas_bar)
    skip_coeff_xt = skip_recified_factor

    skip_coeff_all = skip_alphas, skip_alphas_bar, skip_coeff_xt2x0, skip_coeff_eps2x0, skip_coeff_xt, skip_coeff_x0

    return skip_coeff_all, roi_idxs


def calc_x0_mean_z(input_z, eps, coeff, ii):
    coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff
    
    x0 = coeff_xt2x0[ii] * input_z - coeff_eps2x0[ii] * eps     # calculate predicted x0
    mean_z = coeff_xt[ii] * input_z + coeff_x0[ii] * x0
    return x0, mean_z


@torch.no_grad()
def forward_cfg(model, zt, timesteps, classlabels, cfg_scale, cls):
    classnulls = torch.tensor([cls] * len(zt), device=zt.device)
    
    # discard the predicted noise variance
    cond_eps = model.forward(zt, timesteps, classlabels)[:, :4, :, :]
    uncond_eps = model.forward(zt, timesteps, classnulls)[:, :4, :, :]

    fuse_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
 
    return cond_eps, uncond_eps, fuse_eps


@torch.no_grad()
def weighted_sum(weights, seq_elem):
    out = torch.zeros_like(seq_elem[0]).to(torch.float64)
    for ii, elem in enumerate(seq_elem):
        out += elem * weights[ii]
    out = out.to(dtype=torch.float32)
    return out


def ddpm_skip_sample(num_step=24):
    seed = 0
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    device = "cuda:0"

    coeff_all, skip_idxs = skip_ddpm_coeff(create_ddpm_coeff(), num_step)
    coeff_all = [torch.from_numpy(elem).to(device=device, dtype=torch.float32) for elem in coeff_all]
    alphas, alphas_bar, log_var, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all

    coeff = coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0

    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-ema").to(device)
    vae.eval()

    latent_size = 32

    path = "DiT-XL-2-256x256.pt"
    model = DiT_models["DiT-XL/2"](input_size=latent_size, num_classes=1000, learn_sigma=True).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    class_labels = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], device=device)
    n = len(class_labels)

    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    input_z = copy.deepcopy(z)

    for ii in list(range(0, num_step))[::-1]:
        if ii % 10 == 0:
            print(ii)

        timesteps = torch.ones(n, dtype=torch.int32, device=device) * skip_idxs[ii]

        ret = forward_cfg(model, input_z, timesteps, class_labels, 4.0, 1000)
        cond_eps, uncond_eps, fuse_eps = ret

        fuse_x0, fuse_mean_z = calc_x0_mean_z(input_z, fuse_eps, coeff, ii)

        mean_z = fuse_mean_z
        noise = torch.randn_like(input_z, dtype=torch.float32, device=device)
        output_z = mean_z + torch.exp(0.5 * log_var[ii]) * noise

        input_z = output_z

    samples = vae.decode(input_z / 0.18215).sample
    path = make_path("results/validation/ddpm_%03d__seed_%d__original.png"%(num_step, seed))
    save_image(samples, path, nrow=8, normalize=True, value_range=(-1, 1))

    return


@torch.no_grad()
def ddim_skip_sample(num_step=24):
    seed = 0
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    device = "cuda:0"
    
    coeff_all, skip_idxs = skip_ddim_coeff(create_ddim_coeff(), num_step)
    coeff_all = [torch.from_numpy(elem).to(device=device, dtype=torch.float32) for elem in coeff_all]
    alphas, alphas_bar, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all

    coeff = coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0

    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-ema").to(device)
    vae.eval()

    latent_size = 32

    path = "DiT-XL-2-256x256.pt"
    model = DiT_models["DiT-XL/2"](input_size=latent_size, num_classes=1000, learn_sigma=True).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    class_labels = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], device=device)
    n = len(class_labels)

    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    input_z = copy.deepcopy(z)

    for ii in list(range(0, num_step))[::-1]:
        timesteps = torch.ones(n, dtype=torch.int32, device=device) * skip_idxs[ii]

        ret = forward_cfg(model, input_z, timesteps, class_labels, 4.0, 1000)
        cond_eps, uncond_eps, fuse_eps = ret

        fuse_x0, fuse_mean_z = calc_x0_mean_z(input_z, fuse_eps, coeff, ii)

        mean_z = fuse_mean_z
        output_z = mean_z

        input_z = output_z
    
    samples = vae.decode(input_z / 0.18215).sample
    path = make_path("results/validation/ddim_%03d__seed_%d__original.png"%(num_step, seed))
    save_image(samples, path, nrow=8, normalize=True, value_range=(-1, 1))

    return


def natural_inference(alg_name="ddpm", num_step=24):
    seed = 0
    torch.manual_seed(seed)
    
    device = "cuda:0"
    
    weight_path = "results/%s/%s_%03d.npz"%(alg_name, alg_name, num_step)
    # # Be careful! Make sure that the past_x0_coeff have been normalized to the marginal signal coefficients
    past_x0_coeff, past_eps_coeff, node_coeff = np.load(weight_path).values()
    num_step = past_eps_coeff.shape[0]
    weight_name = os.path.basename(weight_path)[:-4]

    coeff_all, skip_idxs = skip_ddim_coeff(create_ddim_coeff(), num_step)
    coeff_all = [torch.from_numpy(elem).to(device=device, dtype=torch.float32) for elem in coeff_all]
    alphas, alphas_bar, coeff_xt2x0, coeff_eps2x0, coeff_xt, coeff_x0 = coeff_all
    
    # reverse list
    coeff_xt2x0 = coeff_xt2x0.flip(0)
    coeff_eps2x0 = coeff_eps2x0.flip(0)
    
    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-ema").to(device)
    vae.eval()

    latent_size = 32

    path = "DiT-XL-2-256x256.pt"
    model = DiT_models["DiT-XL/2"](input_size=latent_size, num_classes=1000, learn_sigma=True).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    class_labels = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], device=device)
    n = len(class_labels)
    
    seq_x0, seq_eps = [], []
    
    noise = torch.randn(n, 4, latent_size, latent_size, device=device)
    input_z = copy.deepcopy(noise)
    seq_eps.append(noise)

    for kk in range(0, num_step):
        timesteps = torch.ones(n, dtype=torch.int32, device=device) * int(node_coeff[kk, 0])

        ret = forward_cfg(model, input_z, timesteps, class_labels, 4.0, 1000)
        cond_eps, uncond_eps, fuse_eps = ret

        pred_x0 = coeff_xt2x0[kk]*input_z - coeff_eps2x0[kk]*fuse_eps

        seq_x0.append(pred_x0)
          
        curr_noise = torch.randn_like(input_z, dtype=torch.float32, device=device)
        seq_eps.append(curr_noise)
        
        next_x0 = weighted_sum(past_x0_coeff[kk], seq_x0)
        next_eps = weighted_sum(past_eps_coeff[kk], seq_eps)
        output_z = next_x0 + next_eps
         
        input_z = output_z

    samples = vae.decode(input_z / 0.18215).sample
    path = make_path("results/validation/%s__seed_%d__natural.png" % (weight_name, seed))
    save_image(samples, path, nrow=8, normalize=True, value_range=(-1, 1))
  
    return
    

def compare_output_tx():
    ddpm_skip_sample(24)
    ddim_skip_sample(24)
    natural_inference("ddpm", 24)
    natural_inference("ddim", 24)
    return


if __name__ == "__main__":
    """
    This code relies on the pre-trained DiT model(DiT-XL-2-256x256.pt) and its corresponding decoder(sd-vae-ft-ema).
    Please download them and place them in the root directory.
    Once the execution is complete, you can find two types of output results in the results/validation folder:
    one from the original algorithm and the other from the corresponding Natural Inference. You'll observe that there is no difference between them.
    """
    compare_output_tx()