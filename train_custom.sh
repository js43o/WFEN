# ==============
# Train WFEN
# ==============
python train.py --gpus 1 --name 4-9_vq-only_add-skip-2 --model vqwfen \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /data/VCL/Jiseung/datasets/ffhq_custom-aligned --dataset_name clean_ffhq --batch_size 16 --total_epochs 50 \
    --visual_freq 500 --print_freq 50 --save_latest_freq 10000 \
    # --continue_train --load_iter 440000 # 

# ===============
# Train WFENHD
# ===============

# python train.py --gpus 1 --name 4-1_feature_level_vq --model vqwfen \
#     --Gnorm 'in' --g_lr 0.0001 --d_lr 0.0004 --beta1 0.5 --load_size 128 --total_epochs 10 \
#     --Dnorm 'in' --num_D 3 --n_layers_D 3 \
#     --dataroot /4tb/datasets/ffhq_custom-aligned --dataset_name blind_ffhq --batch_size 8 \
#     --visual_freq 1000 --print_freq 10 --save_latest_freq 1000 --no_wavelet #--continue_train

