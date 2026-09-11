#include "OcrLite.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>\n"
                << "Example: ./ocr_demo images/id_card.jpg\n";
        return 1;
    }

    const std::string imgPath = argv[1];
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << imgPath << std::endl;
        return 1;
    }

    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;

    OcrLite ocr;
    ocr.setNumThread(4);

    // Paths relative to the executable (models are copied by CMake)
    bool ok = ocr.initModels(
        "models/ch_PP-OCRv4_det_infer.onnx",
        "models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "models/en_PP-OCRv4_rec_infer.onnx",
        "models/en_dict.txt");

    if (!ok) {
        std::cerr << "Failed to load models. Make sure the four files are in models/\n";
        return 1;
    }

    std::cout << "Models loaded. Running OCR...\n";

    auto t0 = std::chrono::steady_clock::now();
    OcrResult result = ocr.detect(img,
                                    /*padding=*/50,
                                    /*maxSideLen=*/1024,
                                    /*boxScoreThresh=*/0.5f,
                                    /*boxThresh=*/0.3f,
                                    /*unClipRatio=*/1.6f,
                                    /*doAngle=*/true,
                                    /*mostAngle=*/true);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "===== OCR Result =====\n";
    std::cout << "Detected lines: " << result.textBlocks.size() << "\n";
    std::cout << result.strRes << std::endl;

    if (!result.boxImg.empty()) {
        cv::imwrite("result.jpg", result.boxImg);
        std::cout << "Visualization saved to result.jpg\n";
    }

  return 0;
}