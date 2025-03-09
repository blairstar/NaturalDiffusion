
import os, sys
import torch
import glob
import pandas as pd
import numpy as np


def add_vp_noise(samples, alphas_bar, t, seed=200):
    gen = torch.Generator()
    gen.manual_seed(seed)

    noises = torch.randn(samples.shape, generator=gen)
    outputs = samples * np.sqrt(alphas_bar[t]) + noises * np.sqrt(1 - alphas_bar[t])
    return outputs


def add_flow_noise(samples, data_scales, t, seed=200):
    gen = torch.Generator()
    gen.manual_seed(seed)

    noises = torch.randn(samples.shape, generator=gen)
    outputs = samples * data_scales[t] + noises * (1 - data_scales[t])
    return outputs


def get_vp_statistics_tx():
    import numpy as np
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    sigmas = np.sqrt((1 - alphas_bar) / alphas_bar)
    
    # # to do:
    # # specify the feature path
    # # assume feature is stored in the pth format, one file one category, batch_size*feat_dim
    size = 512
    path = "latents_%d/*.pt" % size
    
    paths = glob.glob(path)
    print(paths[0])

    for t in [200, 300, 400, 500, 600, 700, 800, 900]:
        x0_counts, x0_probs = [], []
        xx_counts, xx_probs = [], []
        total_count = 0
        for ii, feat_path in enumerate(paths):
            if ii % 300 == 0:
                print(size, t, ii, os.path.basename(feat_path))

            feats = torch.load(feat_path).to(torch.float32)
            feats = feats.reshape(feats.shape[0], -1)

            samples = add_vp_noise(feats, alphas_bar, t, ii)
            exponent = -1 * torch.cdist(samples, feats, p=2) ** 2 / (2 * sigmas[t] ** 2)

            max_vals = exponent.max(axis=1, keepdim=True)[0]
            ref_dists = (exponent - max_vals).to(dtype=torch.float64)
            exp_vals = torch.exp(ref_dists, out=ref_dists)
            sum_exp_vals = torch.sum(exp_vals, 1, keepdim=True)
            probs = exp_vals / sum_exp_vals
            max_probs = probs.max(axis=1)[0]

            x0_count = (probs.diag() > 0.9).sum().item()
            x0_counts.append(x0_count)
            x0_probs.append(probs.diag().cpu().numpy())

            xx_count = probs.max(axis=1)[0].sum().item()
            xx_counts.append(xx_count)
            xx_probs.append(probs.max(axis=1)[0].cpu().numpy())

            total_count += max_probs.shape[0]

        x0_counts = np.array(x0_counts)
        xx_counts = np.array(xx_counts)
        print(size, t, "%0.4f" % sigmas[t], "%0.4f" % x0_counts.mean(), "%0.4f" % (x0_counts.sum() / total_count))
        print(size, t, "%0.4f" % sigmas[t], "%0.4f" % xx_counts.mean(), "%0.4f" % (xx_counts.sum() / total_count))

        x0_probs = np.concatenate(x0_probs)
        xx_probs = np.concatenate(xx_probs)
        hist_x0, _ = np.histogram(x0_probs, bins=100, range=(0, 1))
        hist_xx, _ = np.histogram(xx_probs, bins=100, range=(0, 1))
        np.savez("hist/vp_%d_%d.npz" % (size, t), hist_x0=hist_x0, hist_xx=hist_xx)

    return


def get_flow_statistics_tx():
    import numpy as np
    data_scales = np.linspace(1, 0.00001, 1000, dtype=np.float64)
    sigmas = (1 - data_scales) / data_scales

    # # to do:
    # # specify the feature path
    # # assume that features are stored in the pth format, one file one category, batch_size*feat_dim
    size = 512
    path = "latents_%d/*.pt" % size
    paths = glob.glob(path)
    print(paths[0])

    for t in [200, 300, 400, 500, 600, 700, 800, 900]:
        x0_counts, x0_probs = [], []
        xx_counts, xx_probs = [], []
        total_count = 0
        for ii, feat_path in enumerate(paths):
            if ii % 300 == 0:
                print(size, t, ii, os.path.basename(feat_path))

            feats = torch.load(feat_path).to(torch.float32)
            feats = feats.reshape(feats.shape[0], -1)

            samples = add_flow_noise(feats, data_scales, t, ii)
            exponent = -1 * torch.cdist(samples, feats, p=2) ** 2 / (2 * sigmas[t] ** 2)

            max_vals = exponent.max(axis=1, keepdim=True)[0]
            ref_dists = (exponent - max_vals).to(dtype=torch.float64)
            exp_vals = torch.exp(ref_dists, out=ref_dists)
            sum_exp_vals = torch.sum(exp_vals, 1, keepdim=True)
            probs = exp_vals / sum_exp_vals
            max_probs = probs.max(axis=1)[0]

            x0_count = (probs.diag() > 0.9).sum().item()
            x0_counts.append(x0_count)
            x0_probs.append(probs.diag().cpu().numpy())

            xx_count = probs.max(axis=1)[0].sum().item()
            xx_counts.append(xx_count)
            xx_probs.append(probs.max(axis=1)[0].cpu().numpy())

            total_count += max_probs.shape[0]

        x0_counts = np.array(x0_counts)
        xx_counts = np.array(xx_counts)
        print(size, t, "%0.4f" % sigmas[t], "%0.4f" % x0_counts.mean(), "%0.4f" % (x0_counts.sum() / total_count))
        print(size, t, "%0.4f" % sigmas[t], "%0.4f" % xx_counts.mean(), "%0.4f" % (xx_counts.sum() / total_count))

        x0_probs = np.concatenate(x0_probs)
        xx_probs = np.concatenate(xx_probs)
        hist_x0, _ = np.histogram(x0_probs, bins=100, range=(0, 1))
        hist_xx, _ = np.histogram(xx_probs, bins=100, range=(0, 1))
        np.savez("hist/flow_%d_%d.npz" % (size, t), hist_x0=hist_x0, hist_xx=hist_xx)

    return


if __name__ == "__main__":
    # get_vp_statistics_tx()
    get_flow_statistics_tx()