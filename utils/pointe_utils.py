import os

import numpy as np
import torch
from loguru import logger
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint
from tqdm.auto import tqdm


def init_from_pointe(prompt, ckpt_version):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("[point-e] creating base model...")
    base_name = "base40M-textvec"
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    logger.debug("[point-e] creating upsample model...")
    upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
    logger.debug("[point-e] downloading base checkpoint...")
    finetuned_ckpt_330k = "pointE_finetuned_with_330kdata.pth"
    finetuned_ckpt_825k = "pointE_finetuned_with_825kdata.pth"
    if '825k' in ckpt_version and os.path.exists(f"./point_e_model_cache/{finetuned_ckpt_825k}"):
        logger.debug("[point-e] load finetuned checkpoint: " + finetuned_ckpt_825k)
        base_model.load_state_dict(
            torch.load(
                os.path.join("./point_e_model_cache", finetuned_ckpt_825k),
                map_location=device,
            )["model_state_dict"]
        )
    elif '330k' in ckpt_version and os.path.exists(f"./point_e_model_cache/{finetuned_ckpt_330k}"):
        logger.debug("[point-e] load finetuned checkpoint: " + finetuned_ckpt_330k)
        base_model.load_state_dict(
            torch.load(
                os.path.join("./point_e_model_cache", finetuned_ckpt_330k),
                map_location=device,
            )["model_state_dict"]
        )
    else:
        base_model.load_state_dict(load_checkpoint(base_name, device))
    logger.debug("[point-e] downloading upsampler checkpoint...")
    upsampler_model.load_state_dict(load_checkpoint("upsample", device))
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=("texts", ""),  # Do not condition the upsampler at all
    )
    # Produce a sample from the model.
    samples = None
    for x in tqdm(
        sampler.sample_batch_progressive(
            batch_size=1, model_kwargs=dict(texts=[prompt])
        )
    ):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = pc.coords
    rgb = np.zeros_like(xyz)
    rgb[:, 0], rgb[:, 1], rgb[:, 2] = (
        pc.channels["R"],
        pc.channels["G"],
        pc.channels["B"],
    )
    return xyz, rgb
