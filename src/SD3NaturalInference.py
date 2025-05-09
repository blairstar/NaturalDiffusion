

from pathlib import Path
root_path = Path(__file__).resolve().parent.parent

import copy
import cv2

from diffusers import StableDiffusion3Pipeline
from PIL import Image
import torch
import numpy as np
import json
import pandas as pd
import os

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def make_path(path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def interleave(seq_img):
    images = torch.stack(seq_img, dim=1).flatten(0, 1)
    return images.contiguous()


def pad(img, length=1):
    img = np.pad(img, ((length, length), (length, length), (0, 0)), mode='constant', constant_values=0)
    return img
    

def save_imgs(imgs, path, size=(256, 256), col=8):
    width, height = size
    row = int(np.ceil(len(imgs)/col))
    canva = np.zeros(((height+2)*row+2, (width+2)*col+2, 3), dtype=np.uint8)
    
    for r in range(row):
        for c in range(col):
            idx = r*col + c
            if idx >= len(imgs):
                continue
            
            img = imgs[idx]
            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
                
            rst, cst = 2+r*(height+2), 2+c*(width+2)
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
            canva[rst:rst+height, cst:cst+width] = img
            
    canva = np.ascontiguousarray(canva)
    cv2.imwrite(path, canva[:, :, ::-1])
    return canva


def euler_weighted_sum(seq_xstarts, cliplen=0):
    acc_xstarts = torch.zeros_like(seq_xstarts[0][1])
    acc_weight = 0
    for weight, xstarts in seq_xstarts[-cliplen:]:
        acc_xstarts += weight*xstarts
        acc_weight += weight
    equiv_xstarts = acc_xstarts/acc_weight
    
    return acc_xstarts, equiv_xstarts


'''
Here, we calculate the coefficient matrix with the following way:
x_N = \epsilon \\
x_{N-1} = \epsilon + (\sigma_{N-1}-\sigma_{N})(\epsilon - x^{N}_0) \\
x_{N-2} = \epsilon + (\sigma_{N-1}-\sigma_{N})(\epsilon - x^{N}_0) + (\sigma_{N-2}-\sigma_{N-1})(\epsilon - x^{N-1}_0) \\
...
where x^{N}_0 is the predict xstart in the N_th discrete time point
This way is slightly different from the one in the paper but produces the same results.
'''
@torch.no_grad()
def sd_euler_natural_inference_tx():
    device, dtype = "cuda", torch.float16
    prompt = "A cat holding a sign that says hello world"
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                    torch_dtype=dtype, local_files_only=True).to(device)

    is_vanilla_update = False
    
    n = 4
    prompts = [prompt]*n
    generator = torch.Generator("cuda")
    generator.manual_seed(10)
    noises = torch.randn(n, 16, 128, 128, device=device, dtype=dtype, generator=generator)

    prompt_emb_info = pipe.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None, negative_prompt="")
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_emb_info

    pipe.scheduler.set_timesteps(28, device=device)
    timesteps, sigmas = pipe.scheduler.timesteps, pipe.scheduler.sigmas
    
    outputs = []
    curr_outputs = copy.deepcopy(noises)
    seq_xstarts = []
    for i, t in enumerate(timesteps[:]):
        print(t.item(), timesteps[i].item(), sigmas[i].item(), -1*(sigmas[i+1]-sigmas[i]).item(), sigmas[i].item()/(1-sigmas[i].item()+1E-5))

        ts = t.expand(curr_outputs.shape[0])

        model_inputs = copy.deepcopy(curr_outputs)
        text_outputs = pipe.transformer(hidden_states=model_inputs, timestep=ts, encoder_hidden_states=prompt_embeds,
                                        pooled_projections=pooled_prompt_embeds, return_dict=False)[0]
        null_outputs = pipe.transformer(hidden_states=model_inputs, timestep=ts, encoder_hidden_states=negative_prompt_embeds,
                                        pooled_projections=negative_pooled_prompt_embeds, return_dict=False)[0]
        
        cliplen = 0
         
        input_xstarts = euler_weighted_sum(seq_xstarts, cliplen)[1] if len(seq_xstarts) > 0 else torch.zeros_like(model_inputs)
        null_xstarts = model_inputs - sigmas[i] * null_outputs
        text_xstarts = model_inputs - sigmas[i] * text_outputs
        fuse_outputs = null_outputs + 7 * (text_outputs - null_outputs)
        fuse_xstarts = model_inputs - sigmas[i] * fuse_outputs
            
        seq_xstarts.append([-1*(sigmas[i+1]-sigmas[i]), fuse_xstarts])        # (predict_xstart weight, predict_xstart)
        
        if is_vanilla_update:   # update next x_t with vanilla euler
            curr_outputs = model_inputs + (sigmas[i+1]-sigmas[i]) * fuse_outputs
        else:                   # update next x_t with natural inference
            curr_outputs = sigmas[i+1] * noises + (1-sigmas[i+1]) * euler_weighted_sum(seq_xstarts)[1]

        output_xstarts = euler_weighted_sum(seq_xstarts, cliplen)[1]
        outputs.append(interleave([input_xstarts, null_xstarts, text_xstarts, fuse_xstarts, output_xstarts]))

    img_alls = []
    outputs = torch.cat(outputs, dim=0)
    for imgs in outputs.split(12):
        imgs = (imgs / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        imgs = pipe.vae.decode(imgs, return_dict=False)[0]      # bug: when size > 16, output abnormal images
        imgs = pipe.image_processor.postprocess(imgs, output_type="pil")
        img_alls.extend(imgs)
    
    # due to the large number of images, this step takes a long time.
    path = make_path(root_path/"results/sd3/euler_seq_clip0.png")
    save_imgs(img_alls, path, (256, 256), 20)

    output_xstarts = (output_xstarts / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    images = pipe.vae.decode(output_xstarts, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images, output_type="pil")

    img_all = np.hstack([np.array(image)[:, :, ::-1] for image in images])
    cv2.imwrite(root_path/"results/sd3/euler_sgl_clip0.png", img_all)
 
    return


def weighted_sum(seq_xstarts, weights=None):
    n = len(seq_xstarts)
    
    acc_weight = 0
    acc_arr = torch.zeros_like(seq_xstarts[0])
    for ii, arr in enumerate(seq_xstarts):
        weight = 1 if weights is None else weights[n-1][ii]
        acc_arr += arr * weight
        acc_weight += weight
    avg_arr = acc_arr/acc_weight
    
    return avg_arr


@torch.no_grad()
def sd_natural_inference_tx():
    device, dtype = "cuda", torch.float16
    prompt = "A cat holding a sign that says hello world"
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                    torch_dtype=dtype, local_files_only=True).to(device)

    n = 4
    prompts = [prompt]*n
    generator = torch.Generator("cuda")
    generator.manual_seed(10)
    noises = torch.randn(n, 16, 128, 128, device=device, dtype=dtype, generator=generator)

    prompt_emb_info = pipe.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None, negative_prompt="")
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_emb_info
    
    num_step = 28
    pipe.scheduler.set_timesteps(num_step, device=device)
    timesteps, sigmas = pipe.scheduler.timesteps, pipe.scheduler.sigmas
    timesteps, sigmas = timesteps.to(device), sigmas.to(device)
    
    dir_path = root_path/"weights"
    weight_names = ["sd3_step_28_weight.csv", "sd3_step_28_weight_sharp.csv"]
    
    for weight_name in weight_names:
        weights = pd.read_csv(os.path.join(dir_path, weight_name), index_col=0).to_numpy()

        outputs = []
        seq_xstarts = []
        idxs = list(range(num_step))
        for ii, kk in enumerate(idxs):
            print(timesteps[kk].item(), sigmas[kk].item(), sigmas[kk].item()/(1-sigmas[kk].item()+1E-5))
            
            ts = timesteps[kk].expand(noises.shape[0])
            sigma = sigmas[kk]
            
            curr_xstarts = weighted_sum(seq_xstarts, weights) if len(seq_xstarts) != 0 else torch.zeros_like(noises)
            
            model_inputs = sigma*noises + (1-sigma)*curr_xstarts
            text_outputs = pipe.transformer(hidden_states=model_inputs, timestep=ts, encoder_hidden_states=prompt_embeds,
                                            pooled_projections=pooled_prompt_embeds, return_dict=False)[0]
            null_outputs = pipe.transformer(hidden_states=model_inputs, timestep=ts, encoder_hidden_states=negative_prompt_embeds,
                                            pooled_projections=negative_pooled_prompt_embeds, return_dict=False)[0]
            
            null_xstarts = model_inputs - sigma * null_outputs
            text_xstarts = model_inputs - sigma * text_outputs
            fuse_xstarts = null_xstarts + 7*(text_xstarts - null_xstarts)
                
            seq_xstarts.append(fuse_xstarts)
            
            output_xstarts = weighted_sum(seq_xstarts, weights)
            
            outputs.append(interleave([curr_xstarts, null_xstarts, text_xstarts, fuse_xstarts, output_xstarts]))

        img_alls = []
        outputs = torch.cat(outputs, dim=0)
        for imgs in outputs.split(12):
            imgs = (imgs / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            imgs = pipe.vae.decode(imgs, return_dict=False)[0]      # bug: when size > 16, output abnormal images
            imgs = pipe.image_processor.postprocess(imgs, output_type="pil")
            img_alls.extend(imgs)

        path = make_path(root_path/"results/sd3/seq_%s.png"%(weight_name[:-4]))
        
        # due to the large number of images, this step takes a long time.
        save_imgs(img_alls, path, (256, 256), 20)

        output_xstarts = (output_xstarts/pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        images = pipe.vae.decode(output_xstarts, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type="pil")
        
        img_all = np.hstack([np.array(image)[:, :, ::-1] for image in images])
        cv2.imwrite(root_path/"results/sd3/sgl_%s.png"%(weight_name[:-4]), img_all)
        
    return


if __name__ == "__main__":
    # sd_euler_natural_inference_tx()
    sd_natural_inference_tx()
