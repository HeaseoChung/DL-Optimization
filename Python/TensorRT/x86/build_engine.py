#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import io
import logging
import torch

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from image_batcher import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(
            np.dtype(self.image_batcher.dtype).itemsize
            * np.prod(self.image_batcher.shape)
        )
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _ = next(self.batch_generator)
            log.info(
                "Calibrating image {} / {}".format(
                    self.image_batcher.image_index, self.image_batcher.num_images
                )
            )
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        # 현존하는 tensorrt plugins 을 IpluginRegistry 에 초기화 또는 등록
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.profile = self.builder.create_optimization_profile()
        self.config.max_workspace_size = 8 * (2**30)  # 8 GB

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, inputs, module, opt_shape_param=None):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        ### Using Onnx path
        if isinstance(module, str):
            module = os.path.realpath(module)
            with open(module, "rb") as f:
                if not self.parser.parse(f.read()):
                    log.error("Failed to load ONNX file: {}".format(module))
                    for error in range(self.parser.num_errors):
                        log.error(self.parser.get_error(error))
                    sys.exit(1)

        ### Using PyTorch Object
        else:
            if opt_shape_param:
                dynamic_opt = {
                    "input": {2: "input_heights", 3: "input_widths"},  # 가변적인 길이를 가진 차원
                    "output": {2: "output_heights", 3: "output_widths"},
                }
            else:
                dynamic_opt = None

            f = io.BytesIO()
            torch.onnx.export(
                module,
                inputs,
                f,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_opt,
            )
            f.seek(0)
            onnx_bytes = f.read()
            self.parser.parse(onnx_bytes)

        ### Input & Output Setting
        self.inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]
        self.outputs = [
            self.network.get_output(i) for i in range(self.network.num_outputs)
        ]

        log.info("Network Description")
        for input in self.inputs:
            self.batch_size = input.shape[0]
            log.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in self.outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

        for input_index, input_tensor in enumerate(self.inputs):
            if opt_shape_param is not None:
                log.info("Creating network as dynamic")
                min_shape = tuple(opt_shape_param[input_index][0][:])
                opt_shape = tuple(opt_shape_param[input_index][1][:])
                max_shape = tuple(opt_shape_param[input_index][2][:])
            else:
                log.info("Creating network as static")
                opt_shape = tuple(input_tensor.shape)
                min_shape = opt_shape
                max_shape = opt_shape
            self.profile.set_shape(
                ["input"][input_index], min_shape, opt_shape, max_shape
            )
        self.config.add_optimization_profile(self.profile)

    def create_engine(
        self,
        engine_path,
        precision="fp16",
        calib_input=None,
        calib_shape=None,
        calib_cache=None,
        calib_num_images=25000,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_shape: The shape for calibration including batch size.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    # list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(
                            calib_input,
                            calib_shape,
                            calib_dtype,
                            max_num_images=calib_num_images,
                            exact_batches=True,
                        )
                    )

        with self.builder.build_engine(self.network, self.config) as engine, open(
            engine_path, "wb"
        ) as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())
