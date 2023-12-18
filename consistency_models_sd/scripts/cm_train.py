"""
Train a diffusion model on images.
"""

import argparse
import torch
from torchvision import transforms

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, DDIMScheduler
from cm import logger
from cm.script_util import (
    cm_train_defaults,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import cm.dist_util as dist
from cm.diffusion import DenoiserSD
from copy import deepcopy
from cm.coco_dataset import COCODataset, InfiniteSampler


def main():
    args = create_argparser().parse_args()

    dist.init()
    logger.configure()
    torch.set_num_threads(40)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size
        
    #############
    # Load data #
    #############

    if args.dataset == 'coco':
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
        dataset = COCODataset(args.dataset_path, subset_name='train2014', transform=transform)
        dataset_sampler = InfiniteSampler(dataset=dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=0)
        data = iter(torch.utils.data.DataLoader(
            dataset=dataset, sampler=dataset_sampler, batch_size=batch_size)
        )
    else:
        raise(f"Unsupported dataset {args.dataset}...")
    
    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    # Load the teacher model for distillation
    logger.log(f"loading the teacher model from {args.teacher_model_path}")
    teacher_pipe = StableDiffusionPipeline.from_pretrained(
        args.teacher_model_path, 
        torch_dtype=torch.float16, 
        variant="fp16", 
    ).to(dist.dev())
    teacher_pipe.scheduler = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
    teacher_pipe.scheduler.final_alpha_cumprod = torch.tensor(1.0) # set boundary condition
    teacher_model = teacher_pipe.unet

    # Create the main pipe
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.teacher_model_path, 
        torch_dtype=torch.float32
    )
    pipe.vae = teacher_pipe.vae # fp16 vae 
    pipe.text_encoder = teacher_pipe.text_encoder # fp16 text encoder
    pipe.scheduler = teacher_pipe.scheduler
    pipe.to(dist.dev())
    model = pipe.unet.train()

    logger.log("creating the target model")
    target_model = deepcopy(model).to(dist.dev())

    # Check that all models have the same parameters
    for dst, src in zip(target_model.parameters(), model.parameters()):
        assert (dst.data == src.data).all()
    
    assert len(list(target_model.buffers())) == len(list(model.buffers())) == len(list(teacher_model.buffers())) == 0 

    # Create SD denoiser
    diffusion = DenoiserSD(
        pipe,
        sigma_data = 0.5,
        loss_norm = args.loss_norm,
        num_timesteps=args.start_scales,
        weight_schedule=args.weight_schedule,
        use_fp16=args.use_fp16
    )
    
    logger.log("training...")
    CMTrainLoop(
        model=model,
        diffusion=diffusion,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_pipe=teacher_pipe,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        guidance_scale=args.guidance_scale,
        use_random_guidance_scales=args.use_random_guidance_scales,
        # Eval fid
        coco_ref_stats_path=args.coco_ref_stats_path,
        inception_path=args.inception_path,
        # COCO prompts for FID evaluation
        coco_max_cnt=args.coco_max_cnt,
        coco_prompt_path=args.coco_prompt_path
    ).run_loop()


def create_argparser():
    defaults = dict(
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=20,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='coco', 
        dataset_path="data/coco",
        coco_prompt_path="data/coco/eval_prompts.csv",
        weight_schedule='uniform',
        teacher_dropout=0.0,
        guidance_scale=8.0,
        use_random_guidance_scales=False,
        coco_max_cnt=5000,
        # Eval fid
        coco_ref_stats_path="data/coco/fid_stats_mscoco256_val.npz",
        inception_path=None,
    )
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
