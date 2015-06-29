#include "gdfmm/gdfmm.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>
#include <queue>
#include <algorithm>
#include <utility>
#include <cassert>

#include <cstdio>

namespace gdfmm {

static inline double sumAt(
                            const cv::Mat &imageI,
                            int x, int y,
                            int windowSize, int *pixCount = 0) {
  assert(imageI.depth() == CV_64F);

  double sum;
  int n;
  Point topLeft, bottomRight, topRight, bottomLeft;
  int windowRadius = windowSize / 2;

  topLeft.x = std::max(0, x - windowRadius);
  topLeft.y = std::max(0, y - windowRadius);

  bottomRight.x = std::min(imageI.cols - 1, x + windowRadius + 1);
  bottomRight.y = std::min(imageI.rows - 1, y + windowRadius + 1);

  topRight.y = topLeft.y;
  topRight.x = bottomRight.x;
  bottomLeft.x = topLeft.x;
  bottomLeft.y = bottomRight.y;

  sum = imageI.at<double>(bottomRight.y, bottomRight.x)
      + imageI.at<double>(topLeft.y, topLeft.x)
      - imageI.at<double>(topRight.y, topRight.x)
      - imageI.at<double>(bottomLeft.y, bottomLeft.x);

  n = (bottomRight.y - topLeft.y) * (bottomRight.x - topLeft.x);
  if (pixCount) *pixCount = n;
  return sum;
}

static inline double meanAt(const cv::Mat &imageI,
                            int x, int y, int windowSize, int *pixCount = 0) {
  int n;
  double sum;
  sum = sumAt(imageI, x, y, windowSize, &n);
  if (pixCount)
    *pixCount = n;
  return sum / n;
}

cv::Mat GuidedFilter(const cv::Mat &object,
                     const cv::Mat &referenceO,
                     cv::Mat *output,
                     int windowSize,
                     float epsilon) {
  cv::Mat objectI;
  cv::Mat reference;
  cv::Mat referenceI;
  cv::Mat reference2I;
  cv::Mat objref;
  cv::Mat objrefI;

  object.convertTo(objref, CV_32F);
  referenceO.convertTo(reference, CV_32F);
  objref = objref.mul(reference);

  cv::integral(object, objectI, CV_64F);
  cv::integral(reference, referenceI, reference2I, CV_64F);
  cv::integral(objref, objrefI, CV_64F);

  printf("%lf %i\n", objref.at<float>(0,0), object.at<uint8_t>(0,0));
  printf("%lf %lf\n", referenceI.at<double>(1,1), reference2I.at<double>(1,1));

  // compute the gradient for each cell
  cv::Mat A(object.rows, object.cols, CV_32F);
  cv::Mat B(object.rows, object.cols, CV_32F);

  // compute the variance in each window, a, and b
  for (int y=0; y<object.rows; y++) {
    for (int x=0; x<object.cols; x++) {
      // compute variance
      int pixCount;
      float variance;
      double ref_sqSum = sumAt(reference2I, x, y, windowSize, &pixCount),
              ref_sum = sumAt(referenceI, x, y, windowSize);
      float ref_mean = static_cast<float>(ref_sum) / pixCount;

      double obj_sum = sumAt(objectI, x, y, windowSize);
      float obj_mean = static_cast<float>(obj_sum) / pixCount;

      double objref_sum = sumAt(objrefI, x, y, windowSize);

      // E(X^2) - E(X)^2
      variance = (static_cast<float>(ref_sqSum) / pixCount)
            - ref_mean * ref_mean;

      // compute a
      A.at<float>(y,x) = 
        ( (objref_sum)/pixCount
            - ref_mean * obj_mean  )
        /
        (variance + epsilon);
      // compute b
      B.at<float>(y,x) = static_cast<float>(obj_sum) / pixCount
          - A.at<float>(y,x) * ref_mean;
    }
  }
  
  cv::Mat AI, BI, result(object.rows, object.cols, object.depth());

  cv::integral(A, AI, CV_64F);
  cv::integral(B, BI, CV_64F);

  for (int y=0; y<object.rows; y++) {
    for (int x=0; x<object.cols; x++) {
      switch (object.depth()) {
        case CV_64F:
          result.at<double>(y,x) = static_cast<double>(meanAt(AI, x, y, windowSize) * reference.at<float>(y, x)
              + meanAt(BI, x, y, windowSize));
          break;
        case CV_32F:
          result.at<float>(y,x) = static_cast<float>(meanAt(AI, x, y, windowSize) * reference.at<float>(y, x)
              + meanAt(BI, x, y, windowSize));
          break;
        case CV_8U:
          result.at<uint8_t>(y,x) = static_cast<uint8_t>(meanAt(AI, x, y, windowSize) * reference.at<float>(y, x)
              + meanAt(BI, x, y, windowSize));
          break;
        case CV_16U:
          result.at<uint16_t>(y,x) = static_cast<uint16_t>(meanAt(AI, x, y, windowSize) * reference.at<float>(y, x)
              + meanAt(BI, x, y, windowSize));
          break;
        default:
          assert(false);
          break;
      }
    }
  }
  if (output) {
    *output = result;
    return result;
  }
  else {
    return result;
  }
}

}  // namespace gdfmm
