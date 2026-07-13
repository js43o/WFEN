export WANDB_API_KEY=90142575dfa8ad97bc4b974e5757895006e41638
# ==============
# Train WaveletRestormer
# ==============
python train.py --gpus 4 --name A_d16_111 --model lightweight_wavelet_restormer \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 112 \
    --Dnorm "none" --num_D 3 --n_layers_D 2 --d_lr 0.0002 \
    --dataroot ../../datasets/blind-ffhq_128px_20set --dataset_name offline_blind_ffhq --batch_size 48 --total_epochs 3 \
    --visual_freq 500 --print_freq 500 --save_latest_freq 10000
