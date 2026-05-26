# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model wavelet_restormer --name 14-12_pix-adv-0.0025 \
    --load_size 128 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/lfw_custom-aligned_validation \
    --pretrain_model_path check_points/14-12_pix-adv-0.0025/latest_net_G.pth \
    --save_as_dir results/14-12_pix-adv-0.0025/lfw_custom-aligned_validation
