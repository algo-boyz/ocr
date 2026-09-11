#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct TextBox {
  std::vector<cv::Point> points;  // 4 points
  float score = 0.f;
};

struct TextLine {
  TextBox box;
  std::string text;
  float score = 0.f;
};

struct OcrResult {
  std::vector<TextLine> textLines;
  double detectTime = 0.0;
  double angleTime  = 0.0;
  double recTime    = 0.0;
  double totalTime  = 0.0;
};
