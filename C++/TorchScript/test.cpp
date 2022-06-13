#include <iostream>
#include <memory>

#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    
    int scale = std::stoi(argv[3]);

    torch::jit::script::Module module;
    try {
        // torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
        module = torch::jit::load(argv[1], torch::Device(torch::kCUDA, 1));
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    
    cv::Mat img_rgb_u8 = cv::imread(argv[2]);
    
    //to tensor
    at::Tensor input = torch::from_blob(img_rgb_u8.data, {img_rgb_u8.rows, img_rgb_u8.cols, 3}, torch::kByte);
    
    input = input.unsqueeze(0);
    input = input.permute({0, 3, 1, 2});
    input = input.div(255);

    input = input.to(torch::kFloat32).to(torch::Device(torch::kCUDA, 1));

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    at::Tensor output = module.forward(inputs).toTensor();

    output = output.permute({0, 2, 3, 1});
    output = output.squeeze();
    output = output.mul(255);
    output = output.clamp(0, 255).to(torch::kU8).to(torch::kCPU);
    output = output.contiguous();

    cv::Mat resultImg(img_rgb_u8.rows * scale, img_rgb_u8.cols * scale, CV_8UC3);
    std::memcpy((void*)resultImg.data, output.data_ptr(), sizeof(torch::kU8) * output.numel());
    cv::imwrite("sr.png", resultImg); 

  std::cout << "Finish\n";
}