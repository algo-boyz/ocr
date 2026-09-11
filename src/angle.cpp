#include "angle.h"
#include <iostream>

bool AngleNet::initModel(const std::string& path, int numThread) {
  this->numThread = numThread;
  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(numThread);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, path.c_str(), opts);
    std::cout << "AngleNet loaded: " << path << std::endl;
    return true;
  } catch (const Ort::Exception& e) {
    std::cerr << "AngleNet init failed: " << e.what() << std::endl;
    return false;
  }
}

bool AngleNet::getAngle(const cv::Mat& src) {
  // Skeleton – real code runs the 0/180 classifier
  return false; // assume upright
}
