export WANDB_API_KEY=90142575dfa8ad97bc4b974e5757895006e41638
# ==============
# Fine-tune Multi-frame WaveBFR model
# ==============
python train.py --gpus 1 --name helo --model multi_frame_wavebfr \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 112 \
    --Dnorm "none" --num_D 3 --n_layers_D 2 --d_lr 0.0002 \
    --dataroot ../../datasets/multipie_crop_patch_v2/train --dataset_name multi_frame_multipie --batch_size 56 --total_epochs 30 \
    --visual_freq 500 --print_freq 500 --save_latest_freq 10000 \
    --pretrain_model_path check_points/breeze/latest_net_G.pth
