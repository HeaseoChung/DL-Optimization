import os
import sys
import logging
import argparse
import torch

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from build_engine import EngineBuilder

from torchvision.models.resnet import resnet50

def main(args):
    # create some regular pytorch model...
    model = resnet50().cuda().eval()
    x = torch.ones((1, 3, 224, 224)).cuda()

    # convert to TensorRT feeding sample data as input
    opt_shape_param = [
        [
            [1, 3, 128, 128],   # min
            [1, 3, 256, 256],   # opt
            [1, 3, 512, 512]    # max
        ]
    ]

    builder = EngineBuilder(args.verbose)
    builder.create_network(x, model, opt_shape_param)
    builder.create_engine(
        args.engine,
        args.precision,
        args.calib_input,
        args.calib_cache,
        args.calib_num_images,
        args.calib_batch_size,
        args.calib_preprocessor,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument(
        "--calib_cache",
        default="./calibration.cache",
        help="The file path for INT8 calibration cache to use, default: ./calibration.cache",
    )
    parser.add_argument(
        "--calib_num_images",
        default=25000,
        type=int,
        help="The maximum number of images to use for calibration, default: 25000",
    )
    parser.add_argument(
        "--calib_batch_size", default=8, type=int, help="The batch size for the calibration process, default: 1"
    )
    parser.add_argument(
        "--calib_preprocessor",
        default="V2",
        choices=["V1", "V1MS", "V2"],
        help="Set the calibration image preprocessor to use, either 'V2', 'V1' or 'V1MS', default: V2",
    )
    args = parser.parse_args()
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not any([args.calib_input, args.calib_cache]):
        parser.print_help()
        log.error("When building in int8 precision, either --calib_input or --calib_cache are required")
        sys.exit(1)
    main(args)

