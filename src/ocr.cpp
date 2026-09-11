#include "ocr.h"
#include "db.h"
#include "angle.h"
#include "crnn.h"
#include <iostream>

OcrLite::OcrLite() = default;
OcrLite::~OcrLite() = default;

void OcrLite::setNumThread(int num) {
  numThread = num;
}

bool OcrLite::initModels(const std::string& detPath,
                         const std::string& clsPath,
                         const std::string& recPath,
                         const std::string& keysPath) {
  dbNet    = std::make_unique<DbNet>();
  angleNet = std::make_unique<AngleNet>();
  crnnNet  = std::make_unique<CrnnNet>();

  bool ok = true;
  ok &= dbNet->initModel(detPath, numThread);
  ok &= angleNet->initModel(clsPath, numThread);
  ok &= crnnNet->initModel(recPath, keysPath, numThread);

  if (ok) {
    std::cout << "All models initialized successfully.\n";
  }
  return ok;
}

OcrResult OcrLite::detect(const cv::Mat& img,
                          int padding,
                          int maxSideLen,
                          float boxScoreThresh,
                          float boxThresh,
                          float unClipRatio,
                          bool doAngle,
                          bool mostAngle) {
  OcrResult result;

  // 1. Detection
  std::vector<TextBox> boxes = dbNet->getTextBoxes(img, maxSideLen,
                                                   boxScoreThresh, boxThresh,
                                                   unClipRatio);
  std::cout << "Detected " << boxes.size() << " text boxes.\n";

  // 2. For each box: crop → (optional) angle → recognize
  for (const auto& box : boxes) {
    // Simple axis-aligned crop for the skeleton (real code does perspective transform)
    cv::Rect rect = cv::boundingRect(box.points);
    rect &= cv::Rect(0, 0, img.cols, img.rows);
    if (rect.width < 5 || rect.height < 5) continue;

    cv::Mat crop = img(rect).clone();

    if (doAngle) {
      // Angle classification (0 / 180)
      // angleNet->getAngle(...)  – left as exercise / use full RapidOcrOnnx
    }

    TextLine line;
    line.box = box;
    line.text = crnnNet->getTextLine(crop);   // CTC decode
    line.score = 0.9f;                        // placeholder
    result.textLines.push_back(line);
  }

  return result;
}
