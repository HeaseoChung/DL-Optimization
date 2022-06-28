import torch
import logging
from build_engine import EngineBuilder
from model_zoo.models import Generator


from model_zoo.models import Generator
from build_engine import EngineBuilder

x = torch.ones((1, 3, 224, 224)).cuda()
use_dynamic_shape = false
use_onnx = false
if use_dynamic_shape:
    opt_shape_param = [
        [
            [1, 3, 128, 128],   # min
            [1, 3, 256, 256],   # opt
            [1, 3, 512, 512]    # max
        ]
    ]
else:
    opt_shape_param = None

if use_onnx:
    model = "Generator_x2_16x64.onnx"
else:
    model = Generator().cuda().eval()

builder = EngineBuilder(args.verbose)
builder.create_network(x, model, opt_shape_param)
builder.create_engine(trt_save_path)
