# export CUDA_VISIBLE_DEVICES=3
# ================================================================================
# Test WFEN on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model dual_feature_restormer --name 14-2_dual_feature-iwt_lf-id_hf-adv_pix-weak-l1_two-level-2 \
    --load_size 128 --dataset_name validation --dataroot /vcl2/Jiseung/datasets/lfw_custom-aligned_validation \
    --pretrain_model_path check_points/14-2_dual_feature-iwt_lf-id_hf-adv_pix-weak-l1_two-level-2/latest_net_G.pth \
    --save_as_dir results/14-2_dual_feature-iwt_lf-id_hf-adv_pix-weak-l1_two-level-2/lfw_custom-aligned_validation
