"""
Based on: https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py and 
          https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
"""
from typing import Optional, Union, List

import time
import numpy as np
import torch
import torch.nn.functional as F
import cm.dist_util as dist

from diffusers import StableDiffusionPipeline


class DenoiserSD:
    def __init__(
        self,
        pipe,
        sigma_data: float = 0.5,
        weight_schedule: str = "uniform",
        loss_norm: str = "l2",
        num_timesteps: int = 50,
        use_fp16: bool = False,
    ):
        self.pipe = pipe
        self.scheduler = pipe.scheduler
        self.sigmas = (1 - self.scheduler.alphas_cumprod) ** 0.5
        self.sigma_data = sigma_data
        self.sigma_max = max(self.sigmas)
        self.sigma_min = min(self.sigmas)
        self.weight_schedule = weight_schedule
        self.loss_norm = loss_norm
        self.num_timesteps = num_timesteps
        self.use_fp16 = use_fp16

        self.generator = torch.Generator(
            device=self.pipe._execution_device
        ).manual_seed(dist.get_rank())

    def consistency_losses(
        self,
        ddp_model,
        teacher_model, 
        target_model, 
        image: Union[
            torch.FloatTensor,
            np.ndarray,
            List[torch.FloatTensor],
            List[np.ndarray],
        ], 
        prompt: Union[str, List[str]],
        guidance_scale: float = 8.0,
        latents: Optional[torch.FloatTensor] = None,
        num_scales: int = 50,
        use_random_guidance_scales: bool = False,
        **kwargs
    ):
        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(prompt, 0, 1, None, None, None)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        device = self.pipe._execution_device
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 or use_random_guidance_scales
        
        with torch.no_grad():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device,
                1, do_classifier_free_guidance
            )
            assert prompt_embeds.dtype == torch.float16

        # 4. Sample timesteps uniformly
        self.scheduler.set_timesteps(num_scales, device=device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device) 
        indices = torch.randint(
            0, num_scales - 1, (batch_size,), generator=self.generator, device=device
        )
        t  = self.scheduler.timesteps[indices]
        t2 = self.scheduler.timesteps[indices + 1]
        
        # 5. Prepare latent variables
        with torch.no_grad():
            latents = self.pipe.prepare_latents(
                image, t, batch_size, 1, prompt_embeds.dtype, device, self.generator
            )
            assert latents.dtype == torch.float16
    
        # 6 Get x_0(x_t) using the distilled model
        assert ddp_model.module.training
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        with torch.cuda.amp.autocast(dtype=torch.float16 if self.use_fp16 else torch.float32):
            distiller_noise_pred = ddp_model(
                latent_model_input,
                torch.cat([t] * 2) if do_classifier_free_guidance else t, 
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            
        if use_random_guidance_scales:
            # Set random guidance_scales from [1, 8] for each sample in a batch
            # TODO: get min and max guidance_scales from kwargs
            guidance_scale = torch.randint(
                1, 9, (batch_size, 1, 1, 1), generator=self.generator, device=device
            ).float()

        if do_classifier_free_guidance:
            distiller_noise_pred_uncond, distiller_noise_pred_text = distiller_noise_pred.chunk(2)
            distiller_noise_pred = distiller_noise_pred_uncond + \
                guidance_scale * (distiller_noise_pred_text - distiller_noise_pred_uncond)

        distiller = scheduler_denoise(self.scheduler, distiller_noise_pred, t, latents)

        # 7 Get x_t-1 using the teacher model
        with torch.no_grad():
            teacher_noise_pred = teacher_model(
                latent_model_input.to(torch.float16),
                torch.cat([t] * 2) if do_classifier_free_guidance else t,
                encoder_hidden_states=prompt_embeds.to(torch.float16),
                return_dict=False,
            )[0].to(torch.float32)

            # perform guidance
            if do_classifier_free_guidance:
                teacher_noise_pred_uncond, teacher_noise_pred_text = teacher_noise_pred.chunk(2)
                teacher_noise_pred = teacher_noise_pred_uncond + \
                    guidance_scale * (teacher_noise_pred_text - teacher_noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents_prev = scheduler_step(self.scheduler, teacher_noise_pred, t, t2, latents)
            
        # 8 Get x_0(x_t-1) using target distilled model
        with torch.no_grad():
            latent_prev_model_input = torch.cat([latents_prev] * 2) if do_classifier_free_guidance else latents_prev
            
            with torch.cuda.amp.autocast(dtype=torch.float16 if self.use_fp16 else torch.float32):
                distiller_target_noise_pred = target_model(
                    latent_prev_model_input,
                    torch.cat([t2] * 2) if do_classifier_free_guidance else t2,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            if do_classifier_free_guidance:
                distiller_target_noise_pred_uncond, distiller_target_noise_pred_text = distiller_target_noise_pred.chunk(2)
                distiller_target_noise_pred = distiller_target_noise_pred_uncond + \
                    guidance_scale * (distiller_target_noise_pred_text - distiller_target_noise_pred_uncond)

            distiller_target = scheduler_denoise(self.scheduler, distiller_target_noise_pred, t2, latents_prev)

        # 9 Compute the loss
        sigmas = self.sigmas.to(dist.dev())[t]
        snrs = get_snr(sigmas)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = torch.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "pseudo-huber":
            d = latents[0].numel()
            c = 0.00054 * d ** 0.5
            diffs = (distiller - distiller_target) ** 2
            loss = (mean_flat(diffs) + c ** 2) ** 0.5 - c
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        return loss, indices


def scheduler_denoise(scheduler, epsilon, timestep, sample):
    assert scheduler.config.prediction_type == "epsilon", \
        f"DiffusionSD does not support prediction type: {scheduler.config.prediction_type}"
    # 1. compute alphas, betas
    dims = sample.ndim
    alpha_prod_t = torch.where(
        timestep > 1, 
        scheduler.alphas_cumprod[timestep], 
        scheduler.final_alpha_cumprod
    ) 
    alpha_prod_t = append_dims(alpha_prod_t, dims)
    beta_prod_t = 1 - alpha_prod_t

    # 2. compute predicted original sample from predicted noise also called
    denoised_sample = (sample - beta_prod_t ** (0.5) * epsilon) / alpha_prod_t ** (0.5)
    return denoised_sample


def scheduler_step(
    scheduler,
    pred_epsilon: torch.FloatTensor,
    timestep: torch.IntTensor, 
    prev_timestep: torch.IntTensor,
    sample: torch.FloatTensor
):
    assert scheduler.config.prediction_type == "epsilon", \
        f"DenoiserSD does not support prediction type: {scheduler.config.prediction_type}"
    assert (prev_timestep >= 0).all() and (timestep > 0).all(), "Timesteps must be non-negative int values"
    dims = sample.ndim

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t = append_dims(alpha_prod_t, dims)
    alpha_prod_t_prev = torch.where(
        prev_timestep > 0, scheduler.alphas_cumprod[prev_timestep], scheduler.final_alpha_cumprod
    ) 
    alpha_prod_t_prev = append_dims(alpha_prod_t_prev, dims)
    beta_prod_t = 1 - alpha_prod_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)
    
    # 3. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

    # 4. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    return prev_sample


@torch.no_grad()
def sample(
    pipe,
    prompt, 
    generator=None, 
    num_inference_steps=50, 
    guidance_scale=8.0,
    num_scales=50,
    sampler='ddim', # ddim or stochastic
    ts=None,
    output_type='pil',
    use_fp16=True
):
    assert isinstance(pipe, StableDiffusionPipeline), f"Does not support the pipeline {type(pipe)}"
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(prompt, height, width, 1, None, None, None)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    device = pipe._execution_device
    
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        1,
        do_classifier_free_guidance
    )
    assert prompt_embeds.dtype == torch.float16

    # 4. Sample timesteps uniformly (first step is 981)
    pipe.scheduler.set_timesteps(num_scales, device=device)
    pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device) 
    if num_inference_steps == num_scales:
        timesteps = pipe.scheduler.timesteps
    elif ts is None:
        step = num_scales / num_inference_steps
        step_ids = torch.arange(0, num_scales, step).to(int)
        timesteps = pipe.scheduler.timesteps[step_ids]
        if sampler == 'stochastic':
            timesteps = torch.cat([timesteps, pipe.scheduler.timesteps[-1:]])
    else:
        timesteps = pipe.scheduler.timesteps[ts]
    
    assert len(timesteps) == num_inference_steps + 1

    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    assert latents.dtype == torch.float16

    if not use_fp16:
        latents = latents.float()
        prompt_embeds = prompt_embeds.float()

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if sampler == 'stochastic':
                latents = scheduler_denoise(pipe.scheduler, noise_pred, t, latents)
                if i == len(timesteps) - 1:
                    break

                next_t = timesteps[i + 1]
                sqrt_alpha_prod = pipe.scheduler.alphas_cumprod[next_t] ** 0.5
                sqrt_one_minus_alpha_prod = (1 - pipe.scheduler.alphas_cumprod[next_t]) ** 0.5

                noise = torch.randn(latents.shape, dtype=latents.dtype, device=device, generator=generator)
                latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
            elif sampler == 'ddim':
                next_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.zeros_like(timesteps[i + 1])
                latents = scheduler_step(pipe.scheduler, noise_pred, t, next_t, latents)
            
            if use_fp16:
                latents = latents.half()
            # call the callback, if provided
            progress_bar.update()

    image = pipe.vae.decode(latents.half() / pipe.vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    return image



def append_dims(x, target_dims):
    """ Appends dimensions to the end of a tensor until it has target_dims dimensions. """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def mean_flat(tensor):
    """ Take the mean over all non-batch dimensions. """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_weightings(weight_schedule, snrs, sigma_data=0.5):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


def get_snr(sigma):
    return (1 - sigma ** 2) / sigma ** 2
