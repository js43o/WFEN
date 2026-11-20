python test_multi_frame.py --gpus 1 --model multi_frame_wavebfr --name wacv_6m_mf \
    --load_size 112 --dataset_name multi_frame_kface --dataroot ../../datasets/kface_crop_patch_v2/test \
    --pretrain_model_path check_points/wacv_6m_mf/latest_net_G.pth \
    --save_as_dir results/wacv_6m_mf_middle-ref/kface_crop_patch_v2 \
    --wavebfr_backbone wavebfr --wavebfr_dim 16 --wavebfr_num_blocks 2 2 2 4 --wavebfr_heads 1 2 4 8 \
    # --calibration_jpg_dir ./calibration_jpg