#include "crnn.h"
#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>

bool CrnnNet::initModel(const std::string& modelPath,
                        const std::string& keysPath,
                        int numThread) {
  this->numThread = numThread;

  // Load character dictionary
  std::ifstream ifs(keysPath);
  if (!ifs.is_open()) {
    std::cerr << "Cannot open dictionary: " << keysPath << std::endl;
    return false;
  }
  std::string line;
  keys.clear();
  keys.emplace_back("blank");          // CTC blank
  while (std::getline(ifs, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    keys.push_back(line);
  }
  std::cout << "Loaded " << keys.size() << " characters from dictionary\n";

  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(numThread);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);
    std::cout << "CrnnNet loaded: " << modelPath << std::endl;
    return true;
  } catch (const Ort::Exception& e) {
    std::cerr << "CrnnNet init failed: " << e.what() << std::endl;
    return false;
  }
}

std::string CrnnNet::getTextLine(const cv::Mat& src) {
  // Skeleton – real code does resize to height 48, normalize, inference, CTC decode
  // For production use the full CrnnNet.cpp from RapidOcrOnnx
  return "[rec-placeholder]";
}
