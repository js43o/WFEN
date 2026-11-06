# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model wavebfr --name 14-6-4_wo-pix-l1 \
    --load_size 128 --dataset_name scface --dataroot ../../datasets/scface_custom-aligned \
    --pretrain_model_path check_points/14-6-4_wo-pix-l1/latest_net_G.pth \
    --save_as_dir results/14-6-4_wo-pix-l1/scface_custom-aligned
