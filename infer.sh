# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python infer.py --gpus 1 --model wavebfr --name 14-6-4_wo-pix-l1 \
    --load_size 112 --pretrain_model_path check_points/14-6-4_wo-pix-l1/latest_net_G.pth \
    --test_img_path input --save_as_dir output
