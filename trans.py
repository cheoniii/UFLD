import argparse

import time

import os
from pathlib import Path

import torch
import tensorrt as trt

from configs.deepdrive import *
from model.model import parsingNet

if __name__ == "__main__":
    ### Ultra-Fast-Lane-Detection Testing
    input_size = (64, 192)
    use_aux = False
    load_image = True
    batch_size = 1
    image_size = (480, 640)
    save_dir = Path("outputs")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # cls_dim = (96, 18, 2)

    weights_path = Path('/home/chehun/dd/Ultra-Fast-Lane-Detection/epoch_99.trt')
    # weights_path = Path('/home/chehun/dd/Ultra-Fast-Lane-Detection/64192.pth')
    model = parsingNet(
        pretrained=False,
        backbone=backbone,
        # cls_dim = cls_dim, 
        cls_dim=(num_grid + 1, cls_num_per_lane, num_lanes),
        use_aux=use_aux
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    model.load_state_dict(compatible_state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1], device=device)

    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)

    onnx_path = weights_path.with_suffix('.onnx')

    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=['input'], output_names=['output'])
    time.sleep(2)

    # ONNX to TensorRT
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('Failed to parse the ONNX model.')

    # Set up the builder config
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16) # FP16
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 2 GB

    serialized_engine = builder.build_serialized_network(network, config)

    with open(onnx_path.with_suffix(".trt"), "wb") as f:
        f.write(serialized_engine)