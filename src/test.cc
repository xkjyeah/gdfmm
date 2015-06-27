#include "gdfmm/gdfmm.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>

int main() {
  cv::Mat rgb = cv::imread("/home/daniel/nyu_label/rgb/r-1294439283.377657-2381571548.png", CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat dep = cv::imread("/home/daniel/nyu_label/rawdepth/r-1294439283.377657-2381571548.png", CV_LOAD_IMAGE_UNCHANGED);

  gdfmm::GDFMM gdfmm(1, 20, 3);
  cv::Mat rv = gdfmm.InPaint(dep, rgb);
  rv *= 5;
  cv::imshow("result", rv);
  cv::waitKey(0);
}

