<a href="https://github.com/HeaseoChung/DL-Optimization/tree/master/Python/TensorRT/x86"><img src="https://img.shields.io/badge/-Documentation-brightgreen"/></a>

# TensorRT for x86
- Official repository of TensorRT for x86 is [HERE](https://github.com/NVIDIA/TensorRT)
- New features are added in this repository

## Contents
- [New Features](#new-features)
- [Usage](#usage)
  * [TensorRT Build](#build-tensorrt-using-pth-or-onnx)
  * [TensorRT Inference](#inference-tensorrt-using-tensorrt)

## New Features
- Build TensorRT using Dynamic shape
- Build TensorRT with PTQ quantization using calibration
- Build TensorRT using Pytorch object
- Inference TensorRT using Dynamic shape

## Usage

### Build tensorRT using pth or onnx

```python3
from model_zoo.models import Generator
from build_engine import EngineBuilder

### Options
use_dynamic_shape = False
use_onnx = False
precision = "fp16" #fp32, fp16, int8 can be selected

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

x = torch.ones((1, 3, 224, 224)).cuda() # Dummy input
builder = EngineBuilder(True)
builder.create_network(x, model, opt_shape_param)


### For fp32 and fp16
if precision == "fp32" || "fp16":
    builder.create_engine(
        trt_save_path,
        precision=precision,
        )

### For int8
elif precision == "int8":
    builder.create_engine(
        trt_save_path,
        precision=precision,
        calib_input=dataset_path,
        calib_shape=calib_shape, # calib_shape = torch.ones((8, 3, 64, 64))
        calib_cache=calib_cache_path_to_save,
    )
else:
    raise ValueError
```

### Inference tensorRT using tensorRT
```python3
import cv2
import numpy as np
from inference_engine import EngineInferencer


trt_engine = EngineInferencer("model_zoo/test_pth.trt",256,256,dynamic=True)

image = cv2.imread("example/Lenna.png")
height, width, _ = image.shape

trt_engine.inputs, trt_engine.outputs, trt_engine.bindings = trt_engine.dynamicBindingProcess(height,width, 2)

pre = np.transpose(image,[2,0,1])
pre = np.expand_dims(pre,axis=0)/255.

sr = trt_engine.do_inference(pre)

post = sr*255
post = np.clip(post,0,255)
post = np.reshape(post, (1,3,height*2, width*2)).astype(np.uint8)
post = post.squeeze(0)
post = np.transpose(post,[1,2,0])

cv2.imwrite("example/Lenna_x2.png", post)
```