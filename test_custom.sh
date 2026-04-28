# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model vqwfen --name 4-7_simple_concat_feature_level_vq \
    --load_size 128 --dataset_name clean_ffhq --dataroot /4tb/datasets/lfw_custom-aligned \
    --pretrain_model_path check_points/4-7_simple_concat_feature_level_vq/iter_340000_net_G.pth \
    --save_as_dir results/4-7_simple_concat_feature_level_vq/lfw_custom-aligned --is_pretrain

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

# python test.py --gpus 1 --model wfenhd --name wfenhd_no_wavelet_blind_ffhq_custom-aligned_128 \
#     --no_wavelet --load_size 128 --dataset_name blind_ffhq --dataroot /4tb/datasets/lfw_custom-aligned \
#     --pretrain_model_path check_points/wfenhd_no_wavelet_blind_ffhq_custom-aligned_128/latest_net_G.pth \
#     --save_as_dir results/wfenhd_no_wavelet_blind_ffhq_custom-aligned_128/lfw_custom-aligned
