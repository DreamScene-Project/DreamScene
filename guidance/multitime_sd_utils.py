import os
from pathlib import Path
from typing import Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionPipeline
from loguru import logger
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image
from transformers import logging

from utils.viz_utils import TextCanvas, lat2rgb, rgb2sat

from .sd_step import ddim_step, pred_original

logging.set_verbosity_error()


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16,
        vram_O,
        t_range=[0.02, 0.98],
        max_t_range=0.98,
        num_train_timesteps=None,
        use_control_net=False,
        textual_inversion_path=None,
        LoRA_path=None,
        guidance_opt=None,
    ):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        model_key = guidance_opt.model_key
        assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = (
            "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key
        )  # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusionPipeline.from_single_file(
                model_key,
                use_safetensors=True,
                torch_dtype=self.precision_t,
                load_safety_checker=False,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key if not is_safe_tensor else base_model_key,
            subfolder="scheduler",
            torch_dtype=self.precision_t,
        )
        self.sche_func = ddim_step

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained(
                controlnet_model_key, torch_dtype=self.precision_t
            ).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            logger.info("load textual inversion in:.{}".format(textual_inversion_path))

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.num_train_timesteps = (
            num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps
        )
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0,))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.stage_range = [400, 850]
        self.stage_range_step = self.stage_range[1] - self.stage_range[0]
        self.jump_range = [175, 225]

        self.stage_refine_t = guidance_opt.stage_refine_t  # 150

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.rgb_latent_factors = torch.tensor(
            [
                # R       G       B
                [0.298, 0.207, 0.208],
                [0.187, 0.286, 0.173],
                [-0.158, 0.189, 0.264],
                [-0.184, -0.271, -0.473],
            ],
            device=self.device,
        )

    def augmentation(self, *tensors):
        augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
            ]
        )

        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c : cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def noise_norm(self, eps):
        return torch.sqrt(torch.square(eps).sum(dim=[1, 2, 3]))

    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step(
        self,
        text_embeddings_all: torch.Tensor,
        pred_rgb: torch.Tensor,
        pred_depth: torch.Tensor,
        pred_alpha: torch.Tensor,
        grad_scale: float = 1,
        use_control_net: bool = False,
        save_folder: Path = None,
        iteration: int = 0,
        stage_step_rate: float = 0,
        resolution: Tuple = (512, 512),
        guidance_opt: omegaconf.dictconfig.DictConfig = None,
        as_latent: bool = False,
        vds: list = [],
        obj_id: str = "",
    ):

        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        if as_latent:
            latents, _ = self.encode_imgs(pred_depth.repeat(1, 3, 1, 1).to(self.precision_t))
        else:
            latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        if self.noise_temp is None:
            self.noise_temp = torch.randn(
                (
                    latents.shape[0],
                    4,
                    resolution[0] // 8,
                    resolution[1] // 8,
                ),
                dtype=latents.dtype,
                device=latents.device,
                generator=self.noise_gen,
            ) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            noise = torch.randn(
                (
                    latents.shape[0],
                    4,
                    resolution[0] // 8,
                    resolution[1] // 8,
                ),
                dtype=latents.dtype,
                device=latents.device,
                generator=self.noise_gen,
            ) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        text_embeddings_all = text_embeddings_all[:, :, ...]
        text_embeddings_all = text_embeddings_all.reshape(
            -1, text_embeddings_all.shape[-2], text_embeddings_all.shape[-1]
        )
        self.guidance_scale = guidance_opt.guidance_scale

        jump_min = self.jump_range[0]
        jump_max = self.jump_range[1]
        max_step = self.stage_range[1] - int(self.stage_range_step * stage_step_rate)
        rand_list = []

        for i in range(4):
            rand_jump = torch.randint(jump_min, jump_max, (1,))[0].to(self.device)
            if not rand_list:
                rand_list.append(rand_jump)
            elif (rand_list[-1] + rand_jump) < max_step:
                rand_list.append(rand_list[-1] + rand_jump)
            else:
                break
        intervals = len(rand_list)
        ratio = [1.0 / intervals for i in range(intervals)]

        with torch.no_grad():
            pred_scores = self.addnoise_with_cfg(
                latents,
                noise,
                rand_list,
                text_embeddings_all,
                guidance_opt.denoise_guidance_scale,
                False,
                eta=guidance_opt.xs_eta,
                pred_with_uncond=False,
            )
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        grads = []
        t_cols, eps_t_cols, prev_latents_noisy_cols = [], [], []
        ratio_index = 0
        t_list = []

        for cur_ind_t, unet_output, cur_noisy_lat in pred_scores:
            if cur_ind_t == 0:
                continue
            t_list.append(str(cur_ind_t.item()))
            cond, uncond, blank = torch.chunk(unet_output, chunks=3)
            pred_noise = uncond + self.guidance_scale * (cond - uncond)
            grad = w(self.alphas[self.timesteps[cur_ind_t]]) * (pred_noise - blank)
            grad = ratio[ratio_index] * torch.nan_to_num(grad_scale * grad)
            grads.append(grad)
            ratio_index += 1
            if iteration % guidance_opt.vis_interval == 0:
                t_cols.append(self.timesteps[cur_ind_t])
                eps_t_cols.append(pred_noise)
                prev_latents_noisy_cols.append(cur_noisy_lat)
        grad = torch.sum(torch.stack(grads, dim=0), dim=0)

        loss = SpecifyGradient.apply(latents, grad)

        if iteration % guidance_opt.vis_interval == 0:
            eta_text = TextCanvas(self.device)
            eta_text.putText("vds=" + ", ".join(vds))
            eta_text.putText(f"max_step={max_step},t_list=" + ",".join(t_list))

            save_path_iter = os.path.join(
                save_folder,
                "{}_iter_{}_step_vd_{}.jpg".format(obj_id, iteration, "_".join(vds)),
            )

            with torch.no_grad():
                pred_x0_pos_cols = []

                for t_, eps_t_, prev_latents_noisy_ in zip(t_cols, eps_t_cols, prev_latents_noisy_cols):
                    pred_x0_pos_cols.append(
                        self.decode_latents(
                            pred_original(self.scheduler, eps_t_, t_, prev_latents_noisy_).type(self.precision_t)
                        )
                    )

                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate(
                    (grad_abs / grad_abs.max()).mean(dim=1, keepdim=True),
                    (resolution[0], resolution[1]),
                    mode="bilinear",
                    align_corners=False,
                ).repeat(1, 3, 1, 1)
                latents_rgb = F.interpolate(
                    lat2rgb(latents, self.rgb_latent_factors),
                    (resolution[0], resolution[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                viz_images = torch.cat(
                    [
                        pred_rgb,
                        pred_depth.repeat(1, 3, 1, 1),
                        pred_alpha.repeat(1, 3, 1, 1),
                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                        latents_rgb,
                        norm_grad,
                    ]
                    + pred_x0_pos_cols
                    + [eta_text.toImage()],
                    dim=0,
                )
                save_image(viz_images, save_path_iter)
        return loss

    def train_step_gt(
        self,
        text_embeddings_all,
        pred_rgb,
        pred_depth=None,
        pred_alpha=None,
        grad_scale=1,
        use_control_net=False,
        save_folder: Path = None,
        iteration=0,
        stage_step_rate=0,
        resolution=(512, 512),
        guidance_opt=None,
        as_latent=False,
        vds=None,
        obj_id="",
        gid=0,
    ):

        if as_latent:
            latents, _ = self.encode_imgs(pred_depth.repeat(1, 3, 1, 1).to(self.precision_t))
        else:
            latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))

        if self.noise_temp is None:
            self.noise_temp = torch.randn(
                (
                    latents.shape[0],
                    4,
                    resolution[0] // 8,
                    resolution[1] // 8,
                ),
                dtype=latents.dtype,
                device=latents.device,
                generator=self.noise_gen,
            ) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            noise = torch.randn(
                (
                    latents.shape[0],
                    4,
                    resolution[0] // 8,
                    resolution[1] // 8,
                ),
                dtype=latents.dtype,
                device=latents.device,
                generator=self.noise_gen,
            ) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        text_embeddings_all = text_embeddings_all[:, :, ...]
        text_embeddings_all = text_embeddings_all.reshape(
            -1, text_embeddings_all.shape[-2], text_embeddings_all.shape[-1]
        )

        self.guidance_scale = guidance_opt.guidance_scale

        jump_min = self.jump_range[0]
        jump_max = self.jump_range[1]
        max_step = self.stage_range[1] - int(self.stage_range_step * stage_step_rate)
        rand_list = []
        for i in range(4):
            rand_jump = torch.randint(jump_min, jump_max, (1,))[0].to(self.device)
            if not rand_list:
                rand_list.append(rand_jump)
            elif (rand_list[-1] + rand_jump) < max_step:
                rand_list.append(rand_list[-1] + rand_jump)
            else:
                break

        with torch.no_grad():
            pred_scores = self.addnoise_with_cfg(
                latents,
                noise,
                rand_list,
                text_embeddings_all,
                guidance_opt.denoise_guidance_scale,
                False,
                eta=guidance_opt.xs_eta,
                pred_with_uncond=False,
            )

        t_cols, eps_t_cols, prev_latents_noisy_cols = [], [], []

        ratio_index = 0
        t_list = []
        for cur_ind_t, unet_output, cur_noisy_lat in pred_scores:
            if cur_ind_t == 0:
                continue
            t_list.append(str(cur_ind_t.item()))
            cond, uncond, blank = torch.chunk(unet_output, chunks=3)
            pred_noise = uncond + self.guidance_scale * (cond - uncond)

            ratio_index += 1
            t_cols.append(self.timesteps[cur_ind_t])
            eps_t_cols.append(pred_noise)
            prev_latents_noisy_cols.append(cur_noisy_lat)
        eta_text = TextCanvas(self.device)
        eta_text.putText("vds=" + ", ".join(vds))
        eta_text.putText(f"max_step={max_step},t_list=" + ",".join(t_list))

        save_path_iter = os.path.join(
            save_folder,
            "{}_iter_{}_step_gid_{}_vd_{}.jpg".format(obj_id, iteration, gid, "_".join(vds)),
        )
        gt_images = None
        with torch.no_grad():
            pred_x0_pos_cols = []
            for t_, eps_t_, prev_latents_noisy_ in zip(t_cols, eps_t_cols, prev_latents_noisy_cols):
                if gt_images is None:
                    gt_images = self.decode_latents(
                        pred_original(self.scheduler, eps_t_, t_, prev_latents_noisy_).type(self.precision_t)
                    )
                    pred_x0_pos_cols.append(gt_images)
                else:
                    pred_x0_pos_cols.append(
                        self.decode_latents(
                            pred_original(self.scheduler, eps_t_, t_, prev_latents_noisy_).type(self.precision_t)
                        )
                    )

            latents_rgb = F.interpolate(
                lat2rgb(latents, self.rgb_latent_factors),
                (resolution[0], resolution[1]),
                mode="bilinear",
                align_corners=False,
            )

            viz_images = torch.cat(
                [
                    pred_rgb,
                    pred_depth.repeat(1, 3, 1, 1),
                    pred_alpha.repeat(1, 3, 1, 1),
                    rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                    latents_rgb,
                ]
                + pred_x0_pos_cols
                + [eta_text.toImage()],
                dim=0,
            )
            save_image(viz_images, save_path_iter)

        return gt_images

    def addnoise_with_cfg(
        self,
        latents,
        noise,
        rand_list,
        text_embeddings=None,
        cfg=1.0,
        is_noisy_latent=False,
        resolution=(512, 512),
        eta=0.0,
        pred_with_uncond=True,
    ):
        """We use DDIM Inversion folloing LucidDreamer here to keep 3D consistency"""
        text_embeddings = text_embeddings.to(self.precision_t)
        unet = self.unet
        ind_prev_t = torch.zeros(1, dtype=torch.long, device=self.device)[0]
        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])
        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat
        pred_scores = []

        for next_ind_t in rand_list:
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(
                self.precision_t
            )
            latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_])
            latent_model_input = latent_model_input.reshape(
                -1,
                4,
                resolution[0] // 8,
                resolution[1] // 8,
            )
            timestep_model_input = (
                self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            )
            unet_output = unet(
                latent_model_input,
                timestep_model_input,
                encoder_hidden_states=text_embeddings,
            ).sample
            cond, uncond, blank = torch.chunk(unet_output, chunks=3)
            pred_scores.append((cur_ind_t, unet_output, cur_noisy_lat))
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t - cur_t
            cur_noisy_lat = self.sche_func(
                self.scheduler, uncond if pred_with_uncond else blank, cur_t, cur_noisy_lat, -delta_t_, eta
            ).prev_sample
            cur_ind_t = next_ind_t
            del unet_output

        cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
        latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_])
        latent_model_input = latent_model_input.reshape(
            -1,
            4,
            resolution[0] // 8,
            resolution[1] // 8,
        )
        timestep_model_input = (
            self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
        )
        unet_output = unet(
            latent_model_input,
            timestep_model_input,
            encoder_hidden_states=text_embeddings,
        ).sample
        pred_scores.append((cur_ind_t, unet_output, cur_noisy_lat))
        del unet_output
        torch.cuda.empty_cache()
        return pred_scores

    def denoise_with_cfg(
        self,
        latents,
        noise,
        rand_list,
        text_embeddings=None,
        cfg=1.0,
        is_noisy_latent=False,
        resolution=(512, 512),
        eta=0.0,
    ):
        text_embeddings = text_embeddings.to(self.precision_t)
        unet = self.unet
        ind_prev_t = rand_list[0]
        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])
        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat
        pred_scores = []
        for next_ind_t in rand_list:
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(
                self.precision_t
            )
            latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_])
            latent_model_input = latent_model_input.reshape(
                -1,
                4,
                resolution[0] // 8,
                resolution[1] // 8,
            )
            timestep_model_input = (
                self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            )
            unet_output = unet(
                latent_model_input,
                timestep_model_input,
                encoder_hidden_states=text_embeddings,
            ).sample
            cond, uncond, blank = torch.chunk(unet_output, chunks=3)
            pred_scores.append((cur_ind_t, unet_output, cur_noisy_lat))
            pred_noise = uncond + cfg * (cond - uncond)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t - cur_t
            print(f"DeNoise From {cur_t} to {next_t}")
            cur_noisy_lat = self.sche_func(self.scheduler, pred_noise, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t
            del unet_output
        cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
        latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_, cur_noisy_lat_])
        latent_model_input = latent_model_input.reshape(
            -1,
            4,
            resolution[0] // 8,
            resolution[1] // 8,
        )
        timestep_model_input = (
            self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
        )
        unet_output = unet(
            latent_model_input,
            timestep_model_input,
            encoder_hidden_states=text_embeddings,
        ).sample
        cond, uncond, blank = torch.chunk(unet_output, chunks=3)
        pred_scores.append((cur_ind_t, unet_output, cur_noisy_lat))
        torch.cuda.empty_cache()
        return pred_scores

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence
