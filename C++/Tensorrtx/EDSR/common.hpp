#ifndef REAL_ESRGAN_COMMON_H_
#define REAL_ESRGAN_COMMON_H_

#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

#include "NvInfer.h"

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
  std::cout << "Loading weights: " << file << std::endl;
  std::map<std::string, Weights> weightMap;

  // Open weights file
  std::ifstream input(file);
  assert(input.is_open() &&
         "Unable to load weight file. please check if the .wts file path is "
         "right!!!!!!");

  // Read number of weight blobs
  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");

  while (count--) {
    Weights wt{DataType::kFLOAT, nullptr, 0};
    uint32_t size;

    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> size;
    wt.type = DataType::kFLOAT;

    // Load blob
    uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
    for (uint32_t x = 0, y = size; x < y; ++x) {
      input >> std::hex >> val[x];
    }
    wt.values = val;

    wt.count = size;
    weightMap[name] = wt;
  }

  return weightMap;
}

ITensor* ResidualBlock(INetworkDefinition* network,
                       std::map<std::string, Weights>& weightMap, ITensor* x,
                       std::string lname) {
  // Body 1 in ResidualBlock
  IConvolutionLayer* body_1 = network->addConvolutionNd(
      *x, 64, DimsHW{3, 3}, weightMap[lname + ".body.0.weight"],
      weightMap[lname + "body.0.bias"]);
  body_1->setStrideNd(DimsHW{1, 1});
  body_1->setPaddingNd(DimsHW{1, 1});

  IActivationLayer* relu_1 = network->addActivation(
      *body_1->getOutput(0), ActivationType::kLEAKY_RELU);

  // TODO check Do I need to Set ReLU as true
  ITensor* x1 = relu_1->getOutput(0);

  // Body 2 in ResidualBlock
  IConvolutionLayer* body_2 = network->addConvolutionNd(
      *x1, 64, DimsHW{3, 3}, weightMap[lname + ".body.2.weight"],
      weightMap[lname + "body.2.bias"]);
  body_2->setStrideNd(DimsHW{1, 1});
  body_2->setPaddingNd(DimsHW{1, 1});

  ITensor* out = body_2->getOutput(0);

  // Elementwise Sum
  float* scval = reinterpret_cast<float*>(malloc(sizeof(float)));
  *scval = 1.0;
  Weights scale{DataType::kFLOAT, scval, 1};
  float* shval = reinterpret_cast<float*>(malloc(sizeof(float)));
  *shval = 0.0;
  Weights shift{DataType::kFLOAT, shval, 1};
  float* pval = reinterpret_cast<float*>(malloc(sizeof(float)));
  *pval = 1.0;
  Weights power{DataType::kFLOAT, pval, 1};

  IScaleLayer* scaled =
      network->addScale(*out, ScaleMode::kUNIFORM, shift, scale, power);
  IElementWiseLayer* ew1 = network->addElementWise(*scaled->getOutput(0), *x,
                                                   ElementWiseOperation::kSUM);
  return ew1->getOutput(0);
}

#endif