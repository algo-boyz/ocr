#pragma once
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

class AngleNet {
public:
  bool initModel(const std::string& path, int numThread);
  // Returns true if the crop should be rotated 180°
  bool getAngle(const cv::Mat& src);

private:
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "AngleNet"};
  std::unique_ptr<Ort::Session> session;
  int numThread = 4;
};
