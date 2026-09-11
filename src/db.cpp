#include "db.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

bool DbNet::initModel(const std::string& path, int numThread) {
  this->numThread = numThread;
  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(numThread);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, path.c_str(), opts);

    // Input / output names (standard for PP-OCR det)
    inputNames  = {"x"};
    outputNames = {"fetch_name_0"};   // may vary – check with Netron if needed

    std::cout << "DbNet loaded: " << path << std::endl;
    return true;
  } catch (const Ort::Exception& e) {
    std::cerr << "DbNet init failed: " << e.what() << std::endl;
    return false;
  }
}

std::vector<TextBox> DbNet::getTextBoxes(const cv::Mat& src,
                                         int maxSideLen,
                                         float boxScoreThresh,
                                         float boxThresh,
                                         float unClipRatio) {
  // Skeleton: real implementation does resize → normalize → inference →
  // DB post-process (threshold, connected components, unclip, minAreaRect).
  // For a production-ready version, copy the full DbNet.cpp from
  // https://github.com/RapidAI/RapidOcrOnnx

  std::vector<TextBox> boxes;
  // Placeholder – returns empty so the demo still runs
  // TODO: implement or paste full post-processing from RapidOcrOnnx
  return boxes;
}
