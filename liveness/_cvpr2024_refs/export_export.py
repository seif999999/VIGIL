# -*- coding: utf-8 -*-
import argparse
import cv2
import torch
import torch.nn as nn
from cv2_transform import transforms
import torchvision.models as models
from nets.utils import get_model
from nets.mobileone import reparameterize_model


parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('--arch', metavar='ARCH', type=str,
                    help='model architecture')
parser.add_argument('--num_classes', default=1000, type=int, metavar='N',
                    help='number of dataset classes number')
parser.add_argument('--weights_path', type=str)
parser.add_argument('--onnx_file', type=str)

args = parser.parse_args()

class SoftMaxNet(nn.Module):
    def __init__(self, basemodel):
        super(SoftMaxNet, self).__init__()
        self.basemodel = basemodel
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.basemodel(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = get_model(args.arch, args.num_classes)

    print("=> loading checkpoint '{}'".format(args.weights_path))
    state_dict = torch.load(args.weights_path, map_location=torch.device('cpu'))
    if 'state_dict_ema' in state_dict and state_dict['state_dict_ema'] is not None:
        state_dict = state_dict['state_dict_ema']
        del state_dict['n_averaged']
        print('using ema weight')
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    else:
        pass

    for k in list(state_dict.keys()):
        if k.startswith('module.module.'):
            state_dict[k[len("module.module."):]] = state_dict[k]
            del state_dict[k]
        elif k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
        else:
            pass
    ret = model.load_state_dict(state_dict, strict=True)
    print(ret)

    if args.arch.startswith('efficientnet'):
        model.set_swish(memory_efficient=False)

    if args.arch.startswith('mobileone'):
        model = reparameterize_model(model)
        print('reparameterize_model for mobileone-s0')

    model = SoftMaxNet(model)
    model.eval()

    trans = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    cvimg = cv2.imread('readme_images/test.jpg')
    image = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    image = trans(image).unsqueeze(0)
    print(image.shape)
    print(type(image))
    dummy_input = image


    input_name = ['input']
    output_name = ['output']

    torch.onnx.export(model,
                     dummy_input,
                     args.onnx_file,
                     verbose=False,
                     opset_version=11, # the ONNX version to export the model to
                     input_names=input_name,
                     output_names=output_name,
                     dynamic_axes={} #for ES
                     # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # for test
                     )

    import onnx
    onnx_model = onnx.load(args.onnx_file)
    onnx.checker.check_model(onnx_model)
    print("==> ONNX check Passed")
    # print(onnx.helper.printable_graph(onnx_model.graph))

    # 验证pytorch模型结果
    probs = model(dummy_input)
    print('pytorch output: {}'.format(probs))

    # 验证onnx模型结果
    import onnxruntime
    session = onnxruntime.InferenceSession(args.onnx_file)
    session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    print(input_name)
    pred = session.run(output_name, {input_name: dummy_input.numpy()})[0]
    print('onnx output: {}'.format(pred))


