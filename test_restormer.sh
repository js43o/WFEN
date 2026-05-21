# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model dual_feature_restormer --name 14-4_dual_feature-iwt_lf-id_hf-adv_pix-weak-l1-vgg_two-level-2_no-dnorm \
    --load_size 128 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/celeba-hq_custom-aligned_validation \
    --pretrain_model_path check_points/14-4_dual_feature-iwt_lf-id_hf-adv_pix-weak-l1-vgg_two-level-2_no-dnorm/latest_net_G.pth \
    --save_as_dir results/14-4_dual_feature-iwt_lf-id_hf-adv_pix-weak-l1-vgg_two-level-2_no-dnorm/celeba-hq_custom-aligned_validation
