python test_multi_frame.py --gpus 1 --model multi_frame_wavebfr --name breeze_122_multi_frame \
    --load_size 112 --dataset_name multi_frame_multipie --dataroot ../../datasets/multipie_crop_patch_v2/test \
    --pretrain_model_path check_points/breeze_122_mf_v1/latest_net_G.pth \
    --save_as_dir results/breeze_122_mf_v1/temp
