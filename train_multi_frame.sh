export WANDB_API_KEY=90142575dfa8ad97bc4b974e5757895006e41638

python train.py --gpus 1 --name wacv_6m_mf_fiqa-ref --model multi_frame_wavebfr \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 112 \
    --Dnorm "none" --num_D 3 --n_layers_D 2 --d_lr 0.0002 \
    --dataroot ../../datasets/kface_crop_patch_v2/train --dataset_name multi_frame_kface --batch_size 36 --total_epochs 20 \
    --visual_freq 500 --print_freq 500 --save_latest_freq 10000 \
    --pretrain_model_path check_points/wacv_6m/latest_net_G.pth \
    --wavebfr_backbone wavebfr --wavebfr_dim 16 --wavebfr_num_blocks 2 2 2 4 --wavebfr_heads 1 2 4 8
