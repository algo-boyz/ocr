#pragma once
#include "types.h"
#include <memory>
#include <string>

class DbNet;
class AngleNet;
class CrnnNet;

class OcrLite {
public:
  OcrLite();
  ~OcrLite();

  void setNumThread(int num);
  bool initModels(const std::string& detPath,
                  const std::string& clsPath,
                  const std::string& recPath,
                  const std::string& keysPath);

  // Full pipeline: det + (optional) cls + rec
  OcrResult detect(const cv::Mat& img,
                   int padding = 50,
                   int maxSideLen = 1024,
                   float boxScoreThresh = 0.5f,
                   float boxThresh = 0.3f,
                   float unClipRatio = 1.6f,
                   bool doAngle = true,
                   bool mostAngle = true);

private:
  int numThread = 4;
  std::unique_ptr<DbNet>   dbNet;
  std::unique_ptr<AngleNet> angleNet;
  std::unique_ptr<CrnnNet>  crnnNet;
};
