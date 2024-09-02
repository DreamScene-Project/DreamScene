import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

def convert_data(data):
    if data is None:
        return None
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return [convert_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: convert_data(v) for k, v in data.items()}
    else:
        raise TypeError(
            "Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting",
            type(data),
        )


def save_image(save_path, img) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = convert_data(img)
    if img.dtype == np.float16 or img.dtype == np.float32:
        img = (img * 255).round().astype("uint8")
    assert img.dtype == np.uint8 or img.dtype == np.uint16
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(save_path), img)
    # print(save_path)


def vis_render_grad(pred_rgb, grad, latents, device, save_path):
    resolution = pred_rgb.shape[2:4]
    rgb_latent_factors = torch.tensor(
        [
            # R       G       B
            [0.298, 0.207, 0.208],
            [0.187, 0.286, 0.173],
            [-0.158, 0.189, 0.264],
            [-0.184, -0.271, -0.473],
        ],
        device=device,
    )
    lat2rgb = lambda x: torch.clip(
        (x.permute(0, 2, 3, 1) @ rgb_latent_factors.to(x.dtype)).permute(0, 3, 1, 2),
        0.0,
        1.0,
    )
    latents_rgb = F.interpolate(
        lat2rgb(latents.clone()),
        (resolution[0], resolution[1]),
        mode="bilinear",
        align_corners=False,
    )
    _pred_rgb = pred_rgb.clone().detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    grad_abs = torch.abs(grad.detach())
    norm_grad = F.interpolate(
        (grad_abs / grad_abs.max()).mean(dim=1, keepdim=True),
        (resolution[0], resolution[1]),
        mode="bilinear",
        align_corners=False,
    ).repeat(1, 3, 1, 1)

    _norm_grad = (
        norm_grad.squeeze(0)
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    print(latents_rgb.shape)
    latents_rgb = latents_rgb.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    viz_imgs = np.concatenate([_pred_rgb, _norm_grad, latents_rgb], axis=1)

    save_image(save_path, viz_imgs)
