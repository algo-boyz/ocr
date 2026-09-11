#pragma once
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

class CrnnNet {
public:
  bool initModel(const std::string& modelPath,
                 const std::string& keysPath,
                 int numThread);
  std::string getTextLine(const cv::Mat& src);

private:
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "CrnnNet"};
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> keys;
  int numThread = 4;
};
