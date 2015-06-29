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

  cv::Mat InPaint2(const cv::Mat &depthImage,
                      const cv::Mat &rgbImageOriginal,
                      float epsilon = 0,
                      float constant = 1,
                      float truncation = 0.05,
                      cv::Mat *output = nullptr);
  private:
  class ExpCache {
    std::unique_ptr<float []> lookupTable;
    int tableSize_;
    
    public:

    ExpCache(float sigma, int tableSize);
    float operator()(int d);
  };

  template <class PredictMethod>
  cv::Mat InPaintBase(const cv::Mat &depthImage,
                      const cv::Mat &rgbImageOriginal,
                      cv::Mat *output = nullptr,
                      const PredictMethod &predict = PredictMethod());
  float BilateralWeight(const Point &p1,
                        const Point &p2,
                        const cv::Mat &rgbImage);
  float PredictDepth(const cv::Mat &depthImage,
                     const cv::Mat &rgbImage,
                     int x, int y);
  float PredictDepth2(const cv::Mat &depthImage,
                     const cv::Mat &rgbImage,
                     int x, int y,
                     float epsilon,
                     float constant,
                     float truncation);
  ExpCache distExpCache_, colorExpCache_;
  unsigned int windowSize_;
};

cv::Mat GuidedFilter(const cv::Mat &object,
                     const cv::Mat &reference,
                     cv::Mat *output,
                     int windowSize,
                     float epsilon);

struct Point {
  int x, y;
};

}
