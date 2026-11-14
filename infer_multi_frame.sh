python infer_multi_frame.py \
    --gpus 1 --model multi_frame_wavebfr --name breeze_24-224_mf \
    --load_size 112 \
    --input-image ./19062421_S001_L10_E01_C10.png \
    --output ./19062421_S001_L10_E01_C10_result.png \
    --pretrain_model_path check_points/breeze_24-224_mf/latest_net_G.pth \
    --diagnostic-dir ./debug/compare \
    --export-fresh-onnx ./debug/breeze_fresh.onnx
