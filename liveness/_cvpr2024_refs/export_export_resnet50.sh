export PYTHONPATH="."
python export/export.py \
    --arch 'resnet50' \
    --weights_path 'full_resnet50.pth' \
    --num_classes 2 \
    --onnx_file 'full_resnet50.onnx'
