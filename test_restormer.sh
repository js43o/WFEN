# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model wavelet_restormer --name air_112 \
    --load_size 112 --dataset_name validation --dataroot ../../datasets/lfw_custom-aligned_validation \
    --pretrain_model_path check_points/air_112/latest_net_G.pth \
    --save_as_dir results/air_112/lfw_custom-aligned_validation
