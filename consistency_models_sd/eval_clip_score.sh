export SAMPLE_DIR=$1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

python evaluations/clip_score.py --prompt_filepath data/coco/eval_prompts.csv --sample_dir $SAMPLE_DIR --batch_size 200