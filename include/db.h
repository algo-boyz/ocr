#pragma once
#include "types.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

class DbNet {
public:
  bool initModel(const std::string& path, int numThread);
  std::vector<TextBox> getTextBoxes(const cv::Mat& src,
                                    int maxSideLen,
                                    float boxScoreThresh,
                                    float boxThresh,
                                    float unClipRatio);

private:
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "DbNet"};
  std::unique_ptr<Ort::Session> session;
  std::vector<const char*> inputNames;
  std::vector<const char*> outputNames;
  int numThread = 4;
};
