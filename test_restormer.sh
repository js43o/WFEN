# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model restormer --name 10_og \
    --load_size 64 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/lfw_custom-aligned_validation \
    --pretrain_model_path check_points/10_og/latest_net_G.pth \
    --save_as_dir results/10_og/lfw_custom-aligned_validation
