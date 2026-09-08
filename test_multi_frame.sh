python test_multi_frame.py --gpus 1 --model multi_frame_wavebfr --name breeze_mf_v3 \
    --load_size 112 --dataset_name multi_frame_kface --dataroot ../../datasets/kface_crop_patch_v2/test \
    --pretrain_model_path check_points/breeze_mf_v3/latest_net_G.pth \
    --save_as_dir results/breeze_mf_v3/kface_crop_patch_v2
