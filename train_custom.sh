# export CUDA_VISIBLE_DEVICES=2
# =================================================================================
# Train WFEN
# =========================================================================
python train.py --gpus 1 --name wfen_no_wavelet_blind_ffhq_custom-aligned_128 --model wfen \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /4tb/datasets/ffhq_custom-aligned --dataset_name blind_ffhq --batch_size 8 --total_epochs 50 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500 --no_wavelet #--continue_train 
