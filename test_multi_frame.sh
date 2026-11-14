python test_multi_frame.py --gpus 1 --model multi_frame_wavebfr --name breeze_16-224_mf \
    --load_size 112 --dataset_name multi_frame_kface --dataroot ../../datasets/kface_crop_patch_v2/test \
    --pretrain_model_path check_points/breeze_16-224_mf/latest_net_G.pth \
    --save_as_dir results/breeze_16-224_mf/kface_crop_patch_v2 \
    --wavebfr_backbone breeze --wavebfr_dim 16 --wavebfr_num_blocks 2 2 4 --wavebfr_heads 1 2 4 \
    # --calibration_jpg_dir ./calibration_jpg