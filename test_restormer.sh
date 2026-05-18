# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model dual_restormer --name 13-6_dual_image-iwt_hf-adv \
    --load_size 128 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/lfw_custom-aligned_validation \
    --pretrain_model_path check_points/13-6_dual_image-iwt_hf-adv/latest_net_G.pth \
    --save_as_dir results/13-6_dual_image-iwt_hf-adv/lfw_custom-aligned_validation
