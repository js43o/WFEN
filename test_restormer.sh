# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model new_restormer --name 12-8_seperated-iwt_hf-vgg-loss \
    --load_size 128 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/celeba-hq_custom-aligned_validation \
    --pretrain_model_path check_points/12-8_seperated-iwt_hf-vgg-loss/latest_net_G.pth \
    --save_as_dir results/12-8_seperated-iwt_hf-vgg-loss/celeba-hq_custom-aligned_validation
