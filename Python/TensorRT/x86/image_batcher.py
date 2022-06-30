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
import torch
import numpy as np
import random
from PIL import Image


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, path, shape, dtype, max_num_images=None, exact_batches=False):
        """
        :param path: The path directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param preprocessor: Set the preprocessor to use, V1 or V2, depending on which network is being used.
        """
        # Find images in the given input path
        path = os.path.realpath(path)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return (
                os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
            )

        if os.path.isdir(path):
            self.images = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if is_image(os.path.join(path, f))
            ]
            self.images.sort()
        elif os.path.isfile(path):
            if is_image(path):
                self.images.append(path)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), path))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape

        # assert len(self.shape) == 4
        if self.shape.size()[3] == 3:
            self.shape = torch.permute(self.shape, (0, 3, 1, 2))  # NHWC -> NCHW

        self.shape = self.shape.size()
        self.batch_size = self.shape[0]

        # assert self.batch_size > 0
        # self.format = None
        # self.width = -1
        # self.height = -1
        # if self.shape[1] == 3:
        #     self.format = "NCHW"
        #     self.height = self.images[0].shape[1]
        #     self.width = self.images[0].shape[2]
        # elif self.shape[3] == 3:
        #     self.format = "NHWC"
        #     self.height = self.images[0].shape[0]
        #     self.width = self.images[0].shape[1]
        # assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0 : self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes cropping, data type casting, and transposing.
        :param image_path: The path to the image on disk to load.
        :return: A numpy array holding the image sample, ready to be contacatenated into the rest of the batch.
        """

        def rand_crop(image):
            width, height = image.size
            crop_w = random.randint(0, width - self.shape[3])
            crop_h = random.randint(0, height - self.shape[2])
            return image.crop(
                (crop_w, crop_h, crop_w + self.shape[3], crop_h + self.shape[2])
            )

        image = Image.open(image_path)
        image = image.convert(mode="RGB")
        image = rand_crop(image)
        image = np.asarray(image, dtype=self.dtype)
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        return image

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding two items per iteration: a numpy array holding a batch of images, and the list of
        paths to the images loaded within this batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i] = self.preprocess_image(image)

            self.batch_index += 1
            yield batch_data, batch_images
