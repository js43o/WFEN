# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python infer.py --gpus 1 --model wavebfr --name wacv_6m \
    --load_size 112 --pretrain_model_path check_points/wacv_6m/latest_net_G.pth \
    --test_img_path temp/suprema_samples --save_as_dir results/wacv_6m/suprema_samples
