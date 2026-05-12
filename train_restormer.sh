export WANDB_API_KEY=90142575dfa8ad97bc4b974e5757895006e41638
# ==============
# Train Restormer
# ==============
python train.py --gpus 1 --name 11-1_query-reuse_64px --model restormer \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 64 \
    --dataroot /vcl2/Jiseung/datasets/ffhq_custom-aligned_filtered --dataset_name blind_ffhq --batch_size 28 --total_epochs 30 \
    --visual_freq 500 --print_freq 100 --save_latest_freq 10000 
