#include "gdfmm/gdfmm.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>

using namespace gdfmm;

void test_inpaint() {
  cv::Mat rgb = cv::imread("/home/daniel/littlechair/0134_color.png", CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat dep = cv::imread("/home/daniel/littlechair/0133_depth.png", CV_LOAD_IMAGE_UNCHANGED);

  gdfmm::GDFMM gdfmm(1, 20, 3);
  cv::Mat rv = gdfmm.InPaint(dep, rgb);
  rv *= 5;
  cv::imshow("result", rv);
  cv::waitKey(0);
}

void test_guided_filter() {
  cv::Mat rgb = cv::imread("/home/daniel/littlechair/0134_color.png", CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat gray;
  cv::Mat black(rgb.rows, rgb.cols, CV_8U);
  black.setTo(0);

  cv::cvtColor(rgb, gray, CV_RGB2GRAY);

  cv::Mat gf;
  GuidedFilter(gray, rgb, &gf, 11, 100);

  cv::imshow("result", gray);
  cv::waitKey(0);

  cv::imshow("result", gf);
  cv::waitKey(0);
}

int main() {
  test_guided_filter();
}

