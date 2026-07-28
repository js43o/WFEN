# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test_single_frame.py --gpus 1 --model wavebfr --name breeze \
    --load_size 112 --dataset_name multi_frame_kface --dataroot ../../datasets/kface_crop_patch_v2/test \
    --pretrain_model_path check_points/breeze/latest_net_G.pth \
    --save_as_dir results/breeze/kface_crop_patch_v2
