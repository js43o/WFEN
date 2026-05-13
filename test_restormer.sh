# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model restormer --name 12-4_seperated-iwt_plus_batch8 \
    --load_size 128 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/lfw_custom-aligned_validation \
    --pretrain_model_path check_points/12-4_seperated-iwt_plus/latest_net_G.pth \
    --save_as_dir results/12-4_seperated-iwt_plus/lfw_custom-aligned_validation
