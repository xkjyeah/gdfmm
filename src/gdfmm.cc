// Copyright 2015 ETH Zurich. All rights reserved
#include "gdfmm/gdfmm.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>
#include <queue>
#include <algorithm>
#include <utility>
#include <iostream>
#include <cassert>
#include <eigen3/Eigen/Eigen>

#include <cstdio>

#define COV_PREDICT_METHOD

namespace gdfmm {
using std::set;
using std::pair;

static float ComputeSpeed(const cv::Mat &gradientStrength,
                          const Point &position);

GDFMM::GDFMM(float sigmaDistance,
        float sigmaColor,
        int windowSize)
  : distExpCache_(sigmaDistance, windowSize),
    colorExpCache_(sigmaColor, 255),
    windowSize_(windowSize) {
  assert(windowSize_ % 2 == 1 && windowSize >= 3);
}

static pair<float, float> ComputeDepthGradient(
                            const cv::Mat &depthImage,
                            int x, int y) {
  float dx = 0, dy = 0;
  int wx = 0, wy = 0;

  if (x > 0 && depthImage.at<float>(y,x) != 0
            && depthImage.at<float>(y,x-1) != 0) {
    dx += depthImage.at<float>(y,x) - depthImage.at<float>(y,x-1);
    wx++;
  }
  if (x+1 < depthImage.cols
            && depthImage.at<float>(y,x+1) != 0
            && depthImage.at<float>(y,x) != 0) {
    dx += depthImage.at<float>(y,x+1) - depthImage.at<float>(y,x);
    wx++;
  }

  if (y > 0 && depthImage.at<float>(y,x) != 0
            && depthImage.at<float>(y-1,x) != 0) {
    dy += depthImage.at<float>(y,x) - depthImage.at<float>(y-1,x);
    wy++;
  }
  if (y+1 < depthImage.rows
            && depthImage.at<float>(y+1,x) != 0
            && depthImage.at<float>(y,x) != 0) {
    dy += depthImage.at<float>(y+1,x) - depthImage.at<float>(y,x);
    wy++;
  }

//  depthGradientX->at<float>(y,x) = (wx > 0) ? dx / wx : 0;
//  depthGradientY->at<float>(y,x) = (wy > 0) ? dy / wy : 0;
  return std::make_pair(
      (wx > 0) ? dx / wx : 0,
      (wy > 0) ? dy / wy : 0
      );
}
//static void ComputeDepthGradients(
//                            const cv::Mat &depthImage,
//                            cv::Mat *depthGradientX,
//                            cv::Mat *depthGradientY) {
//  for (int y=0; y<depthImage.rows; y++) {
//    for (int x=0; x<depthImage.cols; x++) {
//      float dx = 0, dy = 0;
//      int wx = 0, wy = 0;
//
//      if (x > 0 && depthImage.at<float>(y,x) != 0
//                && depthImage.at<float>(y,x-1) != 0) {
//        dx += depthImage.at<float>(y,x) - depthImage.at<float>(y,x-1);
//        wx++;
//      }
//      if (x+1 < depthImage.cols
//                && depthImage.at<float>(y,x+1) != 0
//                && depthImage.at<float>(y,x) != 0) {
//        dx += depthImage.at<float>(y,x+1) - depthImage.at<float>(y,x);
//        wx++;
//      }
//
//      if (y > 0 && depthImage.at<float>(y,x) != 0
//                && depthImage.at<float>(y-1,x) != 0) {
//        dy += depthImage.at<float>(y,x) - depthImage.at<float>(y-1,x);
//        wy++;
//      }
//      if (y+1 < depthImage.rows
//                && depthImage.at<float>(y+1,x) != 0
//                && depthImage.at<float>(y,x) != 0) {
//        dy += depthImage.at<float>(y+1,x) - depthImage.at<float>(y,x);
//        wy++;
//      }
//
//      depthGradientX->at<float>(y,x) = (wx > 0) ? dx / wx : 0;
//      depthGradientY->at<float>(y,x) = (wy > 0) ? dy / wy : 0;
//    }
//  }
//}


cv::Mat GDFMM::InPaint(const cv::Mat &depthImage,
                      const cv::Mat &rgbImageOriginal,
                      cv::Mat *output) {
  return InPaintBase(depthImage,
                      rgbImageOriginal,
                      output,
                      [this] (const cv::Mat &dI,
                          const cv::Mat &rgbI,
                          int x, int y) {
                        return PredictDepth(dI, rgbI, x, y);
                      });
}

cv::Mat GDFMM::InPaint2(const cv::Mat &depthImage,
                      const cv::Mat &rgbImageOriginal,
                      float epsilon,
                      float constant,
                      float truncation,
                      cv::Mat *output) {
  return InPaintBase(depthImage,
                      rgbImageOriginal,
                      output,
                      [this, epsilon, constant, truncation]
                      (const cv::Mat &dI,
                          const cv::Mat &rgbI,
                          int x, int y) {
                        return PredictDepth2(dI, rgbI,
                                             x, y,
                                             epsilon, constant,
                                             truncation);
                      });
}

template <class PredictMethod>
cv::Mat GDFMM::InPaintBase(const cv::Mat &depthImageOriginal,
                const cv::Mat &rgbImage,
                cv::Mat *output,
                const PredictMethod &predict) {
  assert(rgbImage.cols == depthImageOriginal.cols &&
          rgbImage.rows == depthImageOriginal.rows);
  cv::Mat depthImage;
  //cv::Mat result(depthImageOriginal.rows, depthImageOriginal.cols,
  //                CV_32F);

  // depth gradient
//  cv::Mat depthGradientX(depthImageOriginal.rows, depthImageOriginal.cols, CV_32F);
//  cv::Mat depthGradientY(depthImageOriginal.rows, depthImageOriginal.cols, CV_32F);
  
  // Cannot use Sobel, because depth is sometimes unknown
  depthImageOriginal.convertTo(depthImage, CV_32F);
// ComputeDepthGradients(depthImage, &depthGradientX, &depthGradientY);

  // gradient image, then (Gaussian blur)
  // resize rgb to depth image (specifically for Tango device)
  cv::Mat rgbGradientX(rgbImage.rows, rgbImage.cols, CV_32F);
  cv::Mat rgbGradientY(rgbImage.rows, rgbImage.cols, CV_32F);
  cv::Mat tmp;

  cv::GaussianBlur(rgbImage, tmp, cv::Size(5,5), 1, 1);
  cv::Sobel(tmp, rgbGradientX, CV_32F, 1, 0, 3);
  cv::Sobel(tmp, rgbGradientY, CV_32F, 0, 1, 3);

  cv::Mat rgbGradientStrength(rgbImage.rows, rgbImage.cols, CV_32FC3);

  for (int y=0; y<rgbImage.rows; y++) {
    for (int x=0; x<rgbImage.cols; x++) {
      for (int c=0; c<3; c++) {
        rgbGradientStrength.at<cv::Vec3f>(y, x)[c] = 
            rgbGradientX.at<cv::Vec3f>(y, x)[c] * rgbGradientX.at<cv::Vec3f>(y, x)[c] +
            rgbGradientY.at<cv::Vec3f>(y, x)[c] * rgbGradientY.at<cv::Vec3f>(y, x)[c];
      }
    }
  }

  // build the priority queue
  typedef pair<float, Point> BandItem;
  auto compare = [](const BandItem &p1, const BandItem &p2) -> bool { return p1.first < p2.first; };
  std::priority_queue<BandItem, std::vector<BandItem>, decltype(compare)>
          narrowBand(compare);

  // initialize narrowBand
  for (int y=0; y<depthImage.rows; y++) {
    for (int x=0; x<depthImage.cols; x++) {
      if (depthImage.at<float>(y, x) != 0) {
        narrowBand.emplace(0, Point{x, y});
      }
    }
  }

  // propagate
  while (narrowBand.size() > 0) {
    float speed;
    Point position;

    std::tie(speed, position) = narrowBand.top();
    narrowBand.pop();
    
    // use 4-neighbour
    const Point neighbours[] {
      {0,1},{1,0},{-1,0},{0,-1}
    };
    for (const Point &d: neighbours) {
      Point neighbour{position.x + d.x, position.y + d.y};
      if (!(neighbour.x >= 0 &&
          neighbour.y >= 0 &&
          neighbour.x < depthImage.cols &&
          neighbour.y < depthImage.rows))
        continue;

      if (depthImage.at<float>(neighbour.y, neighbour.x) == 0) {
        depthImage.at<float>(neighbour.y, neighbour.x) =
              predict(depthImage,
                      rgbImage,
                      neighbour.x, neighbour.y);

        //printf("predicted (%d,%d) = %f\n", neighbour.y, neighbour.x, depthImage.at<float>(neighbour.y, neighbour.x));

        float T = ComputeSpeed(rgbGradientStrength, neighbour);

        narrowBand.emplace(T, neighbour);
      }
    }
  }

  if (output) {
    depthImage.convertTo(*output, CV_16U);
    return *output;
  }
  else {
    depthImage.convertTo(depthImage, CV_16U);
    return depthImage;
  }
}

float GDFMM::BilateralWeight(const Point &p1,
                      const Point &p2,
                      const cv::Mat &rgbImage) {
  assert(rgbImage.depth() == CV_8U);
  cv::Vec3f c1 = rgbImage.at<uint8_t>(p1.y, p1.x);
  cv::Vec3f c2 = rgbImage.at<uint8_t>(p2.y, p2.y);
      //printf("%f\n", distExpCache_(p1.x - p1.x) );
      //printf("%f\n", distExpCache_(p2.y - p2.y) );
      //printf("%f\n", colorExpCache_(c1[0] - c2[0]) );
      //printf("%f\n", colorExpCache_(c1[1] - c2[1]) );
      //printf("%f\n", colorExpCache_(c1[2] - c2[2]));
  return
      distExpCache_(p1.x - p1.x) *
      distExpCache_(p2.y - p2.y) *
      colorExpCache_(c1[0] - c2[0]) *
      colorExpCache_(c1[1] - c2[1]) *
      colorExpCache_(c1[2] - c2[2]);
}

float GDFMM::PredictDepth(const cv::Mat &depthImage,
                         const cv::Mat &rgbImage,
                         int x, int y) {
  assert(depthImage.cols == rgbImage.cols);
  assert(depthImage.rows == rgbImage.rows);

  float sumWeights = 0;
  float sumValues = 0;
  int windowRadius = windowSize_ / 2;

  for (int n = std::max(0, y - windowRadius);
       n <= std::min(depthImage.rows - 1, static_cast<int>(y + windowRadius));
       n++) {
    for (int m = std::max(0, x - windowRadius);
         m <= std::min(depthImage.cols - 1, static_cast<int>(x + windowRadius));
         m++) {
      float depth = depthImage.at<float>(n, m);

      if (depth == 0) // invalid
        continue;

      float weight = BilateralWeight(Point{x,y}, Point{m,n}, rgbImage);

      float gX, gY;
      std::tie(gX, gY) = ComputeDepthGradient(depthImage, m, n);
      float gradientTerm = gX * (x-m) + gY * (y-n); 

      //printf("for (%d %d), neighbourhood (%d %d) %f %f %f\n", x,y, m,n, weight, depth, gradientTerm);

      sumValues += weight * (depth + gradientTerm);
      sumWeights += weight;
    }
  }
  
  assert(sumWeights != 0);
  return sumValues / sumWeights;
}

/**
 * An alternative implementation for situations where the missing areas
 * are much larger. It uses the covariance matrix within the window instead.
 *
 * Because it uses regularized least squares, maybe you want to scale
 * your RGB image to [0,1] first
 * **/
float GDFMM::PredictDepth2(const cv::Mat &depthImage,
                         const cv::Mat &rgbImage,
                         int x, int y,
                         float epsilon,
                         float constant,
                         float truncation) {
  assert(depthImage.cols == rgbImage.cols);
  assert(depthImage.rows == rgbImage.rows);
  assert(depthImage.depth() == CV_32F);

  float sumWeights = 0;
  float sumValues = 0;
  int windowRadius = windowSize_ / 2;

  // Count known pixels in the window
  int lowerY = std::max(0, y - windowRadius);
  int lowerX = std::max(0, x - windowRadius);
  int upperY = std::min(depthImage.rows - 1, static_cast<int>(y + windowRadius));
  int upperX = std::min(depthImage.cols - 1, static_cast<int>(x + windowRadius));

  int num_known = 0;
  for (int n = lowerY; n <= upperY; n++) {
    for (int m = lowerX; m <= upperX; m++) {
      if (depthImage.at<float>(n, m) == 0) {
        continue;
      }
      else {
        num_known++;
      }
    }
  }

  if (num_known <= 3) {
    return 0;
    throw "Too few known values. Try densifying your depth image first, "
        "or increasing the window size.";
  }

  // Build the array
  Eigen::Array<float, Eigen::Dynamic, 4> X(num_known, 4);
  Eigen::Array<float, Eigen::Dynamic, 1> Y(num_known);
  int index = 0;
  for (int n = lowerY; n <= upperY; n++) {
    for (int m = lowerX; m <= upperX; m++) {
      if (depthImage.at<float>(n, m) == 0) {
        continue;
      }
      else {
        X(index, 0) = rgbImage.at<cv::Vec3b>(n, m)[0];
        X(index, 1) = rgbImage.at<cv::Vec3b>(n, m)[1];
        X(index, 2) = rgbImage.at<cv::Vec3b>(n, m)[2];
        X(index, 3) = constant;
        Y(index) = depthImage.at<float>(n, m);
        index++;
      }
    }
  }
  
  // Find mean
  Eigen::Array<float, 1, 4> mX = X.colwise().mean();
  Eigen::Array<float, 1, 1> mY = Y.colwise().mean();

  // Find covariance
  Eigen::Matrix4f cov = X.transpose().matrix() * X.matrix();
  Eigen::Matrix4f reg;
  reg << epsilon, 0, 0, 0,
         0, epsilon, 0, 0,
         0, 0, epsilon, 0,
         0, 0, 0, epsilon;

  cov = cov + reg;

  // make prediction
  Eigen::Vector4f beta;
  Eigen::Array<float, 1, 4> xy = (X.colwise() * Y).colwise().sum();
  beta = cov.ldlt().solve(xy.matrix().transpose());
//  Eigen::Array<float, 1, 4> xy = (X.colwise() * Y).colwise().mean();
//  beta = cov.ldlt().solve(xy.matrix().transpose());
  
  Eigen::Vector4f test(rgbImage.at<cv::Vec3b>(y, x)[0],
            rgbImage.at<cv::Vec3b>(y, x)[1],
            rgbImage.at<cv::Vec3b>(y, x)[2],
            constant);
  assert(rgbImage.depth() == CV_8U);
  assert(rgbImage.channels() == 3);
//  std::cout << "Y" << std::endl
//            << Y.transpose() << std::endl;
//  std::cout << "X" << std::endl
//            << X.transpose() << std::endl;
//  std::cout << "beta: " << beta << std::endl;
//  std::cout << "test: " << test << std::endl;
//  std::cout << "cov: " << cov << std::endl;
//  std::cout << "xy: " << xy << std::endl;
//  std::cout << num_known << std::endl;
//  std::cout << beta.dot(test) << std::endl;
  float prediction = beta.dot(test);
  assert(!isnan(prediction));

  // constrain the results to within a sane range
//  float meanY = Y.mean();
//  Eigen::Matrix<float, Eigen::Dynamic, 1> Y_mY = (Y - meanY).matrix();
//  float varY = Y_mY.transpose() * Y_mY;
//  varY /= (num_known - 1);
//  float stdY = sqrt(varY);
//
//  prediction = std::max(meanY - 2 * stdY, prediction);
//  prediction = std::min(meanY + 2 * stdY, prediction);
  float minY = Y.minCoeff();
  float maxY = Y.maxCoeff();
  float rangeY = maxY - minY;
  prediction = std::max(minY - rangeY * truncation, prediction);
  prediction = std::min(maxY + rangeY * truncation, prediction);

  return prediction;
}

static float ComputeSpeed(const cv::Mat &gradientStrength,
                          const Point &position)
{
  return -1.0f / (1 + gradientStrength.at<float>(position.y, position.x));
}


};
