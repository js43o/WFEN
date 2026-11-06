# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model wfen --name wfen \
    --load_size 128 --dataset_name scface --dataroot ../../datasets/scface_custom-aligned \
    --pretrain_model_path check_points/00_wfen/latest_net_G.pth \
    --save_as_dir results/wfen/scface_custom-aligned
