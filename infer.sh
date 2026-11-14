# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python infer.py --gpus 1 --model wavebfr --name wacv_ver \
    --load_size 112 --pretrain_model_path check_points/wacv_ver/latest_net_G.pth \
    --test_img_path results/multiframe_kface/lr_reference --save_as_dir results/wacv_ver/kface_lr_reference
