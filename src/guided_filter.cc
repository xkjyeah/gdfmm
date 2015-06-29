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

#include <eigen3/Eigen/Eigen>
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

static inline Eigen::Vector3d sumAt3(
                            const cv::Mat &imageI,
                            int x, int y,
                            int windowSize, int *pixCount = 0) {
  assert(imageI.depth() == CV_64F);

  Eigen::Vector3d sum;
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

#define SUM_OF_CHANNEL(i) \
          (imageI.at<cv::Vec3d>(bottomRight.y, bottomRight.x)[i] \
        + imageI.at<cv::Vec3d>(topLeft.y, topLeft.x)[i] \
        - imageI.at<cv::Vec3d>(topRight.y, topRight.x)[i] \
        - imageI.at<cv::Vec3d>(bottomLeft.y, bottomLeft.x)[i] )

  sum(0) = SUM_OF_CHANNEL(0);
  sum(1) = SUM_OF_CHANNEL(1);
  sum(2) = SUM_OF_CHANNEL(2);

#undef SUM_OF_CHANNEL

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

/* implementation for 3-channel references */
cv::Mat GuidedFilter3(const cv::Mat &objectO,
                     const cv::Mat &referenceO,
                     cv::Mat *output,
                     int windowSize,
                     float epsilon) {
  using std::unique_ptr;
  assert(referenceO.channels() == 3);
  cv::Mat object;
  cv::Mat objectI;
  cv::Mat reference;
  cv::Mat referenceI;
  unique_ptr<cv::Mat[]> reference2I;
  cv::Mat objref;
  cv::Mat objrefI;

  objectO.convertTo(object, CV_32F);
  referenceO.convertTo(reference, CV_32F);
  reference.copyTo(objref);
  // objref = objref.mul(reference);
  for (int y=0; y<object.rows; y++) {
    for (int x=0; x<object.cols; x++) {
      objref.at<cv::Vec3f>(y,x)[0] *= object.at<float>(y,x);
      objref.at<cv::Vec3f>(y,x)[1] *= object.at<float>(y,x);
      objref.at<cv::Vec3f>(y,x)[2] *= object.at<float>(y,x);
    }
  }

  cv::integral(object, objectI, CV_64F);
  cv::integral(reference, referenceI, CV_64F);
  cv::integral(objref, objrefI, CV_64F);

  // covariance for reference2I
  int size2I = reference.channels() * reference.channels();
  reference2I.reset(new cv::Mat[reference.channels() * reference.channels()]);
  for (int i=0; i<3; i++) {
    for (int j=i; j<3; j++) {
      int index = 3*i + j;
      // multiply pixels in channels together
      cv::Mat &ref2 = reference2I[index];
      ref2.create(reference.rows + 1, reference.cols + 1,
                                CV_64F);
      // initialize
      for (int y=0; y<ref2.rows; y++) {
        ref2.at<double>(y,0) = 0;
      }
      for (int x=0; x<ref2.cols; x++) {
        ref2.at<double>(0,x) = 0;
      }
      // compute
      for (int y=1; y<ref2.rows; y++) {
        for (int x=1; x<ref2.cols; x++) {
          // need hacky access
          float *ref_data = reinterpret_cast<float*>(reference.data);

          double pix_value = ref_data[(y-1)*reference.channels()*reference.cols +
                                      (x-1)*reference.channels() +
                                      i] * 
                             ref_data[(y-1)*reference.channels()*reference.cols +
                                      (x-1)*reference.channels() +
                                      j]; 

          ref2.at<double>(y,x) = pix_value
                        - ref2.at<double>(y-1, x-1)
                        + ref2.at<double>(y, x-1)
                        + ref2.at<double>(y-1, x);
        }
      }
    }
  }

  // compute the gradient for each cell
  cv::Mat A(object.rows, object.cols, CV_32FC3);
  cv::Mat B(object.rows, object.cols, CV_32FC1);

  // compute the variance in each window, a, and b
  for (int y=0; y<object.rows; y++) {
    for (int x=0; x<object.cols; x++) {
      // compute variance
      int pixCount;
      float variance;

      Eigen::Vector3d ref_sum = sumAt3(referenceI, x, y, windowSize, &pixCount);
      double ref_sqSum[9]; // compute Σrr, Σrg, etc.
      double cov[9];
      for (int i=0; i<3; i++) {
        for (int j=i; j<3; j++) {
          ref_sqSum[3*i + j] = sumAt(reference2I[3*i + j], x, y, windowSize, &pixCount);
          
          // compute entry in cov matrix
          cov[3*i + j] = ref_sqSum[3*i + j] / (pixCount - 1) -
                    (ref_sum[i] * ref_sum[j] / pixCount / (pixCount - 1));
        }
      }
      Eigen::Vector3f ref_mean = ref_sum.cast<float>() / pixCount;

      double obj_sum = sumAt(objectI, x, y, windowSize);
      float obj_mean = static_cast<float>(obj_sum) / pixCount;

      Eigen::Vector3d objref_sum = sumAt3(objrefI, x, y, windowSize);

      // E(X^2) - E(X)^2
      Eigen::Matrix3f covariance;
      covariance << cov[0] + epsilon, cov[1], cov[2],
                    cov[1], cov[4] + epsilon, cov[5],
                    cov[2], cov[5], cov[8] + epsilon;

      // compute a
      Eigen::Vector3f Av = covariance.ldlt().solve( (objref_sum/pixCount).cast<float>() - ref_mean*obj_mean );
      float Bv = obj_sum / pixCount - Av.dot(ref_mean);

      A.at<cv::Vec3f>(y,x)[0] = Av[0];
      A.at<cv::Vec3f>(y,x)[1] = Av[1];
      A.at<cv::Vec3f>(y,x)[2] = Av[2];

      B.at<float>(y,x) = Bv;

//      As[y * object.cols + x] = covariance.lldt().solve( (objref_sum/pixCount) - ref_mean*obj_mean ).cast<float>();
//      // compute b
//      Bs[y * object.cols + x] = obj_sum / pixCount
//                                - As[y*object.cols + x].T().cast<float>() * ref_mean;
    }
  }
  
  cv::Mat AI, BI, result(object.rows, object.cols, CV_32F);

  cv::integral(A, AI, CV_64F);
  cv::integral(B, BI, CV_64F);

  for (int y=0; y<object.rows; y++) {
    for (int x=0; x<object.cols; x++) {
      int count;
      Eigen::Vector3f sumA = sumAt3(AI, x, y, windowSize, &count).cast<float>();
      Eigen::Map<Eigen::Vector3f> ref(&reference.at<cv::Vec3f>(y,x)[0]);

      result.at<float>(y,x) = sumA.dot(ref) / count + meanAt(BI, x, y, windowSize);
    }
  }
  result.convertTo(result, objectO.depth());
  
  if (output) {
    *output = result;
    return result;
  }
  else {
    return result;
  }
}
cv::Mat GuidedFilter(const cv::Mat &object,
                     const cv::Mat &referenceO,
                     cv::Mat *output,
                     int windowSize,
                     float epsilon) {
  if (object.rows != referenceO.rows ||
      object.cols != referenceO.cols) {
    throw "Images have different size";
  }
  if (referenceO.channels() == 3) {
    return GuidedFilter3(object, referenceO, output, windowSize, epsilon);
  }
  else if (referenceO.channels() != 1) {
    throw "Wrong number of channels";
  }

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
