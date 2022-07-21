#include <unistd.h>

#include <cmath>

#include "common.hpp"
#include "cuda_utils.h"
#include "logging.h"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "utils.h"

#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

static const int PRECISION_MODE = 16;  // fp32 : 32, fp16 : 16
static const bool VISUALIZATION = true;
static const int INPUT_H = 512;
static const int INPUT_W = 512;
static const int INPUT_C = 3;
static const int OUT_SCALE = 2;
static const int OUTPUT_SIZE =
    INPUT_C * INPUT_H * OUT_SCALE * INPUT_W * OUT_SCALE;
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";
static Logger gLogger;

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder,
                          IBuilderConfig* config, DataType dt,
                          std::string& wts_name) {
  INetworkDefinition* network = builder->createNetworkV2(0U);

  // Create input tensor of shape {INPUT_H, INPUT_W, INPUT_C} with name
  // INPUT_BLOB_NAME
  ITensor* data =
      network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_H, INPUT_W, INPUT_C});
  assert(data);
  std::map<std::string, Weights> weightMap = loadWeights(wts_name);

  // Custom preprocess (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
  Preprocess preprocess{maxBatchSize, INPUT_C, INPUT_H, INPUT_W};
  IPluginCreator* preprocess_creator =
      getPluginRegistry()->getPluginCreator("preprocess", "1");
  IPluginV2* preprocess_plugin = preprocess_creator->createPlugin(
      "preprocess_plugin", (PluginFieldCollection*)&preprocess);
  IPluginV2Layer* preprocess_layer =
      network->addPluginV2(&data, 1, *preprocess_plugin);
  preprocess_layer->setName("preprocess_layer");
  ITensor* prep = preprocess_layer->getOutput(0);

  // conv_Head
  IConvolutionLayer* head = network->addConvolutionNd(*prep, 64, DimsHW{3, 3},
                                                      weightMap["head.weight"],
                                                      weightMap["head.bias"]);
  head->setStrideNd(DimsHW{1, 1});
  head->setPaddingNd(DimsHW{1, 1});
  head->setName("head");
  ITensor* feat = head->getOutput(0);

  // conv_Body
  ITensor* body_feat = ResidualBlock(network, weightMap, feat, "body.0");
  for (int idx = 1; idx < 16; idx++) {
    body_feat = ResidualBlock(network, weightMap, body_feat,
                              "body." + std::to_string(idx));
  }

  // Elementwise Sum with feature from head and body
  IElementWiseLayer* ew1 =
      network->addElementWise(*feat, *body_feat, ElementWiseOperation::kSUM);
  feat = ew1->getOutput(0);

  // upsample
  IConvolutionLayer* tail_0 = network->addConvolutionNd(
      *feat, (64 * (std::pow(OUT_SCALE, 2))), DimsHW{3, 3},
      weightMap["tail.0.weight"], weightMap["tail.0.bias"]);
  tail_0->setStrideNd(DimsHW{1, 1});
  tail_0->setPaddingNd(DimsHW{1, 1});

  IShuffleLayer* shuffle1 = network->addShuffle(*tail_0->getOutput(0));
  Dims dm;
  dm.nbDims = 5;

  dm.d[0] = 64;
  dm.d[1] = OUT_SCALE;
  dm.d[2] = OUT_SCALE;
  dm.d[3] = INPUT_H;
  dm.d[4] = INPUT_W;

  shuffle1->setReshapeDimensions(dm);
  IShuffleLayer* shuffle2 = network->addShuffle(*shuffle1->getOutput(0));

  Permutation perm;
  perm.order[0] = 0;
  perm.order[1] = 3;
  perm.order[2] = 1;
  perm.order[3] = 4;
  perm.order[4] = 2;

  Dims dm2;
  dm2.nbDims = 3;
  dm2.d[0] = 64;
  dm2.d[1] = INPUT_H * OUT_SCALE;
  dm2.d[2] = INPUT_W * OUT_SCALE;

  shuffle2->setFirstTranspose(perm);
  shuffle2->setReshapeDimensions(dm2);
  IActivationLayer* relu = network->addActivation(*shuffle2->getOutput(0),
                                                  ActivationType::kLEAKY_RELU);

  IConvolutionLayer* tail_3 = network->addConvolutionNd(
      *relu->getOutput(0), INPUT_C, DimsHW{3, 3}, weightMap["tail.3.weight"],
      weightMap["tail.3.bias"]);
  tail_3->setStrideNd(DimsHW{1, 1});
  tail_3->setPaddingNd(DimsHW{1, 1});
  ITensor* out = tail_3->getOutput(0);

  // Custom postprocess (RGB -> BGR, NCHW->NHWC, *255, ROUND, uint8)
  Postprocess postprocess{maxBatchSize, out->getDimensions().d[0],
                          out->getDimensions().d[1], out->getDimensions().d[2]};
  IPluginCreator* postprocess_creator =
      getPluginRegistry()->getPluginCreator("postprocess", "1");
  IPluginV2* postprocess_plugin = postprocess_creator->createPlugin(
      "postprocess_plugin", (PluginFieldCollection*)&postprocess);
  IPluginV2Layer* postprocess_layer =
      network->addPluginV2(&out, 1, *postprocess_plugin);
  postprocess_layer->setName("postprocess_layer");

  ITensor* final_tensor = postprocess_layer->getOutput(0);
  final_tensor->setName(OUTPUT_BLOB_NAME);
  network->markOutput(*final_tensor);

  // Build engine
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

  if (PRECISION_MODE == 16) {
    std::cout << "==== precision f16 ====" << std::endl << std::endl;
    config->setFlag(BuilderFlag::kFP16);
  } else {
    std::cout << "==== precision f32 ====" << std::endl << std::endl;
  }

  std::cout << "Building engine, please wait for a while..." << std::endl;
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // Don't need the network any more
  delete network;

  // Release host memory
  for (auto& mem : weightMap) {
    free((void*)(mem.second.values));
  }

  return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream,
                std::string& wts_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  ICudaEngine* engine =
      build_engine(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);

  assert(engine != nullptr);

  // Serialize the engine
  (*modelStream) = engine->serialize();

  // Close everything down
  delete engine;
  delete builder;
  delete config;
}

void doInference(IExecutionContext& context, cudaStream_t& stream,
                 void** buffers, uint8_t* output, int batchSize) {
  // infer on the batch asynchronously, and DMA output back to host
  context.enqueue(batchSize, buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, buffers[1],
                             batchSize * OUTPUT_SIZE * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine,
                std::string& img_dir) {
  if (argc < 4) return false;
  if (std::string(argv[1]) == "-s" && argc == 4) {
    wts = std::string(argv[2]);
    engine = std::string(argv[3]);
  } else if (std::string(argv[1]) == "-d" && argc == 4) {
    engine = std::string(argv[2]);
    img_dir = std::string(argv[3]);
  } else {
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  std::string wts_name = "";
  std::string engine_name = "";
  std::string img_dir;

  if (!parse_args(argc, argv, wts_name, engine_name, img_dir)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./edsr -s [.wts] [.engine] // serialize model to plan file"
              << std::endl;
    std::cerr << "./edsr -d [.engine] ../samples  // deserialize plan file and "
                 "run inference"
              << std::endl;
    return -1;
  }

  // create a model using the API directly and serialize it to a stream
  if (!wts_name.empty()) {
    IHostMemory* modelStream{nullptr};
    APIToModel(BATCH_SIZE, &modelStream, wts_name);
    assert(modelStream != nullptr);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
      std::cerr << "could not open plan output file" << std::endl;
      return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()),
            modelStream->size());
    delete modelStream;
    return 0;
  }

  // deserialize the .engine and run inference
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    return -1;
  }
  char* trtModelStream = nullptr;
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  trtModelStream = new char[size];
  assert(trtModelStream);
  file.read(trtModelStream, size);
  file.close();

  std::vector<std::string> file_names;
  if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    std::cerr << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  IRuntime* runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
  assert(engine != nullptr);
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;
  assert(engine->getNbBindings() == 2);
  void* buffers[2];
  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
  assert(inputIndex == 0);
  assert(outputIndex == 1);

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * INPUT_C * INPUT_H *
                                                  INPUT_W * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&buffers[outputIndex],
                        BATCH_SIZE * OUTPUT_SIZE * sizeof(uint8_t)));

  std::vector<uint8_t> input(BATCH_SIZE * INPUT_H * INPUT_W * INPUT_C);
  std::vector<uint8_t> outputs(BATCH_SIZE * OUTPUT_SIZE);

  // Create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
  for (int f = 0; f < (int)file_names.size(); f++) {
    for (int b = 0; b < BATCH_SIZE; b++) {
      cv::Mat img = cv::imread(img_dir + "/" + file_names[f]);
      if (img.empty()) continue;
      memcpy(input.data() + b * INPUT_H * INPUT_W * INPUT_C, img.data,
             INPUT_H * INPUT_W * INPUT_C);
    }

    CUDA_CHECK(cudaMemcpyAsync(
        buffers[inputIndex], input.data(),
        BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t),
        cudaMemcpyHostToDevice, stream));

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, (void**)buffers, outputs.data(), BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
  }

  cv::Mat frame = cv::Mat(INPUT_H * OUT_SCALE, INPUT_W * OUT_SCALE, CV_8UC3,
                          outputs.data());
  cv::imwrite(file_names[0], frame);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(buffers[inputIndex]));
  CUDA_CHECK(cudaFree(buffers[outputIndex]));
  // Destroy the engine
  delete context;
  delete engine;
  delete runtime;
}