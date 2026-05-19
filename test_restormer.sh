# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model dual_restormer --name 13-11_dual_image-iwt_lf-id_hf-adv_pix-l1-vgg_two-level \
    --load_size 128 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/celeba-hq_custom-aligned_validation \
    --pretrain_model_path check_points/13-11_dual_image-iwt_lf-id_hf-adv_pix-l1-vgg_two-level/latest_net_G.pth \
    --save_as_dir results/13-11_dual_image-iwt_lf-id_hf-adv_pix-l1-vgg_two-level/celeba-hq_custom-aligned_validation
