#pragma once

#include <opencv2/core/core.hpp>
#include <memory>

namespace gdfmm {

struct Point;

class GDFMM {
  public:
  GDFMM(float sigmaDistance = 2.0f,
        float sigmaColor = 10.0f,
        int windowSize = 11);
  cv::Mat InPaint(const cv::Mat &depthImage,
                const cv::Mat &rgbImageOriginal,
                cv::Mat *output = nullptr);
  private:
  class ExpCache {
    std::unique_ptr<float []> lookupTable;
    int tableSize_;
    
    public:

    ExpCache(float sigma, int tableSize);
    float operator()(int d);
  };

  float BilateralWeight(const Point &p1,
                        const Point &p2,
                        const cv::Mat &rgbImage);
  float PredictDepth(const cv::Mat &depthImage,
                         const cv::Mat &depthGradientX,
                         const cv::Mat &depthGradientY,
                         const cv::Mat &rgbImage,
                         int x, int y);
  ExpCache distExpCache_, colorExpCache_;
  unsigned int windowSize_;
};

struct Point {
  int x, y;
};

}
