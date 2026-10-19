export PYTHONPATH="."
python export/export.py \
    --arch 'mobilenet_v3_small' \
    --weights_path 'mobilenet_v3_small.pth' \
    --num_classes 2 \
    --onnx_file 'mobilenet_v3_small.onnx'
