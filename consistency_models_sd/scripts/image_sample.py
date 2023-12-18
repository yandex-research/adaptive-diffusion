import os
import argparse
import torch
from torchvision.transforms import ToPILImage
import numpy as np
from tqdm import tqdm
from diffusers import  StableDiffusionPipeline, DDIMScheduler
from cm import logger
from cm.diffusion import sample
from cm.script_util import add_dict_to_argparser
import cm.dist_util as dist
from cm.coco_dataset import prepare_coco_prompts


def main():
    args = create_argparser().parse_args()

    dist.init()
    logger.configure()
    torch.set_num_threads(16)

    # Load the teacher model for distillation
    logger.log(f"loading the original SD model from {args.teacher_model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.teacher_model_path, 
        torch_dtype=torch.float16, 
        variant="fp16", 
    ).to(dist.dev())
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.final_alpha_cumprod = torch.tensor(1.0) # set boundary condition

    if args.sampler == "stochastic":
        if len(args.ts) > 0:
            ts = tuple(int(x) for x in args.ts.split(","))
        else:
            logger.warn(f"Sampling timesteps are not provided. The steps will be selected uniformly.")
            ts = None
    else:
        ts = None

    # Load pretrained checkpoint
    state_dict = torch.load(args.pretrained_model_path)
    pipe.unet.load_state_dict(state_dict)
    if args.compile_model:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    dist.barrier()
    
    local_images = []
    local_text_idxs = []
    logger.info(f"Generating coco samples...")
    rank_batches, rank_batches_index = prepare_coco_prompts(
        args.coco_prompt_path, bs=args.batch_size, max_cnt=args.coco_max_cnt
    )
    generator = torch.Generator(device="cuda").manual_seed(dist.get_rank())
    # for cnt, mini_batch in enumerate(tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
    for cnt, mini_batch in enumerate(tqdm(rank_batches, unit='batch')):
        text = list(mini_batch)
        image = sample(
            pipe,
            text, 
            generator=generator, 
            num_inference_steps=args.num_inference_steps, 
            guidance_scale=args.guidance_scale,
            sampler=args.sampler,
            ts=ts,
            output_type='pil'
        )
        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.from_numpy(np.array(image[text_idx]))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

    local_images = torch.stack(local_images).to(dist.dev())
    local_text_idxs = torch.tensor(local_text_idxs).to(dist.dev())

    gathered_images = [torch.zeros_like(local_images) for _ in range(dist.get_world_size())]
    gathered_text_idxs = [torch.zeros_like(local_text_idxs) for _ in range(dist.get_world_size())]
    
    dist.all_gather(gathered_images, local_images)  # gather not supported with NCCL
    dist.all_gather(gathered_text_idxs, local_text_idxs) 
        
    if dist.get_rank() == 0:
        gathered_images = np.concatenate(
            [images.cpu().numpy() for images in gathered_images], axis=0
        )
        gathered_text_idxs = np.concatenate(
            [text_idxs.cpu().numpy() for text_idxs in gathered_text_idxs], axis=0
        )
        save_dir = os.path.join(logger.get_dir(), f"steps_{args.num_inference_steps}")
        os.makedirs(save_dir, exist_ok=True)
        for image, global_idx in zip(gathered_images, gathered_text_idxs):
            ToPILImage()(image).save(os.path.join(save_dir, f"{global_idx}.jpg"))
    # Done.
    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        batch_size=50,
        use_fp16=False,
        guidance_scale=8.0,
        dataset='coco', 
        dataset_path="data/coco",
        coco_prompt_path="data/coco/eval_prompts.csv",
        coco_max_cnt=5000,
        # Models
        pretrained_model_path='results/ema_model.pt',
        teacher_model_path='runwayml/stable-diffusion-v1-5',
        # Eval fid
        coco_ref_stats_path="data/coco/fid_stats_mscoco256_val.npz",
        inception_path=None,
        num_inference_steps=5,
        sampler='stochastic',
        ts='',
        compile_model=False, # available for torch version > 2.0 
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
