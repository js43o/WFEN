export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model wfen --name wfen_blind_ffhq \
    --load_size 128 --dataset_name blind_ffhq --dataroot /vcl2/Jiseung/datasets/lfw_custom-aligned \
    --pretrain_model_path check_points/wfen_blind_ffhq_custom-aligned_128/iter_200000_net_G.pth \
    --save_as_dir results/blind_ffhq_custom-aligned_128/lfw_custom-aligned

# ----------------- calculate PSNR/SSIM scores ----------------------------------
# python psnr_ssim.py
# ------------------------------------------------------------------------------- 

# ----------------- calculate LPIPS/VIF scores ----------------------------------
# python vif_lpips/lpips_2dirs.py --use_gpu
# python vif_lpips/VIF.py
# ------------------------------------------------------------------------------- 

# ----------------- calculate Parmas/FLOPS scores -------------------------------
# python calc_flops.py
# ------------------------------------------------------------------------------- 
