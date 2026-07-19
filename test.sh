# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model wavebfr --name wacv_ver \
    --load_size 112 --dataset_name validation --dataroot ../../datasets/lfw_custom-aligned_validation_112_easy \
    --pretrain_model_path check_points/wacv_ver/latest_net_G.pth \
    --save_as_dir results/wacv_ver/lfw_custom-aligned_validation_112_easy
