# Consistency Distillation for Stable Diffusion
The implementation of [Consistency Distillation](https://arxiv.org/abs/2303.01469) for Stable Diffusion. This code is based on [consistency_models](https://github.com/openai/consistency_models). 

The pretrained model distilled from StableDiffusion v1.5 on LAION is [here](https://storage.yandexcloud.net/yandex-research/adaptive-diffusion/cd_sd_v1.5_bs512_w8_ema0.9999.pt).

### Dataset

The released model is pretrained on the LAION2B subset that contains ~80M filtered examples. Since LAION requires significant resources to host and download the data, we provide the training/finetuning example on the COCO2014 dataset:
[coco2014.tar.gz](https://storage.yandexcloud.net/yandex-research/adaptive-diffusion/coco2014.tar.gz) - contains the original COCO2014 train set. 

For FID and CLIP score evaluation, we provide a random subset of 30K text prompts from the COCO2014 validation set and precomputed InceptionV3 statistics for FID calculation:
[eval_coco.tar.gz](https://storage.yandexcloud.net/yandex-research/adaptive-diffusion/eval_coco.tar.gz)

### Evaluate the pretrained CD model

1. Download the dataset for evaluation:\
&nbsp;&nbsp; ```bash data/download_coco_for_evaluation.sh```
2. Download the pretrained ema model: \
&nbsp;&nbsp; ```bash pretrained/download_pretrained_model.sh``` 
3. Generate samples with the pretrained model: ```bash sample.sh```
4. Evaluate FID score: ```bash eval_fid_score.sh <path-to-samples>```
4. Evaluate CLIP score: ```bash eval_fid_score.sh <path-to-samples>```

### Train on COCO2014 from scratch

1. Download the dataset:\
&nbsp;&nbsp; ```bash data/download_coco_for_training.sh```
2. Run: ```distill_coco.sh``` 

### Finetune the LAION checkpoint on COCO2014

1. Download the dataset:\
&nbsp;&nbsp; ```bash data/download_coco_for_training.sh```
2. Download the full LAION checkpoint: \
&nbsp;&nbsp; ```bash pretrained/download_full_checkpoint.sh``` 
3. Run: ```bash distill_coco.sh ``` 
