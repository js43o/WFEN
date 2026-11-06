python test_multi_frame.py --gpus 1 --model multi_frame_wavebfr --name wacv_3m_mf \
    --load_size 112 --dataset_name multi_frame_kface --dataroot ../../datasets/kface_crop_patch_v2/test \
    --pretrain_model_path check_points/wacv_3m_mf/latest_net_G.pth \
    --save_as_dir results/wacv_3m_mf/kface_crop_patch_v2
