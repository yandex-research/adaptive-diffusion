export OPENAI_LOGDIR="${PWD}/results"
echo $OPENAI_LOGDIR

export CUDA_VISIBLE_DEVICES=5,7
export OMP_NUM_THREADS=1

python -m torch.distributed.run --standalone --nproc_per_node=2 --master-addr=0.0.0.0:1206 scripts/cm_train.py \
   --target_ema_mode fixed \
   --dataset coco \
   --dataset_path data/coco \
   --start_ema 0.95 \
   --scale_mode fixed \
   --start_scales 50 \
   --total_training_steps 200 \
   --loss_norm l2 \
   --lr_anneal_steps 0 \
   --teacher_model_path runwayml/stable-diffusion-v1-5 \
   --ema_rate 0.999,0.9999,0.9999432189950708 \
   --global_batch_size 32 \
   --microbatch 16 \
   --lr 0.00001 \
   --use_fp16 True \
   --weight_decay 0.0 \
   --save_interval 100 \
   --weight_schedule uniform \
   --coco_max_cnt 5000