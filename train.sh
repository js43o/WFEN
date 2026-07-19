export WANDB_API_KEY=90142575dfa8ad97bc4b974e5757895006e41638
# ==============
# Train WaveBFR model
# ==============
python train.py --gpus 1 --name cloud --model wavebfr \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 112 \
    --Dnorm "none" --num_D 3 --n_layers_D 2 --d_lr 0.0002 \
    --dataroot ../../datasets/ffhq_custom-aligned --dataset_name blind_ffhq --batch_size 64 --total_epochs 50 \
    --visual_freq 500 --print_freq 500 --save_latest_freq 10000
