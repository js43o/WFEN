export WANDB_API_KEY=90142575dfa8ad97bc4b974e5757895006e41638
# ==============
# Train WaveletRestormer
# ==============
python train.py --gpus 1 --name 14-2_dual_feature-iwt_lf-id_hf-adv_pix-weak-l1_two-level-2 --model dual_feature_restormer \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --Dnorm 'in' --num_D 3 --n_layers_D 2 --d_lr 0.0002 \
    --dataroot /vcl2/Jiseung/datasets/blind-ffhq_128px_20set --dataset_name offline_blind_ffhq --batch_size 56 --total_epochs 4 \
    --visual_freq 500 --print_freq 500 --save_latest_freq 10000
