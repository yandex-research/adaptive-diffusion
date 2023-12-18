export SAMPLE_DIR=$1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

python evaluations/fid_score.py --ref_stats_path data/coco/fid_stats_mscoco256_val.npz --sample_dir $SAMPLE_DIR --batch_size 200