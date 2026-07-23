# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test_single_frame.py --gpus 1 --model wavebfr --name breeze_122 \
    --load_size 112 --dataset_name multi_frame_multipie --dataroot ../../datasets/multipie_crop_patch_v2/test \
    --pretrain_model_path check_points/breeze_122/latest_net_G.pth \
    --save_as_dir results/breeze_122/multipie_crop_patch_v2
