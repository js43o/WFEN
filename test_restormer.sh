# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model wavelet_restormer --name 15-2_wo-gated-ffn-exp-3.5 \
    --load_size 128 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/celeba-hq_custom-aligned_validation \
    --pretrain_model_path check_points/15-2_wo-gated-ffn-exp-3.5/latest_net_G.pth \
    --save_as_dir results/15-2_wo-gated-ffn-exp-3.5/celeba-hq_custom-aligned_validation
