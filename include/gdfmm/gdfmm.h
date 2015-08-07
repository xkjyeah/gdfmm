#pragma once

#include <opencv2/core/core.hpp>
#include <memory>

/** @file */
namespace gdfmm {

struct Point;

/** \brief Class to hold the settings and caches for
 * guided depth enhancement.
 *
 * This class assumes either 3-channel 8-bit input image,
 * or a 1-channel image.
 *
 * Depth values of zero are assumed to be missing and require
 * inpainting. Inpainting is performed with information from
 * pixels in the neighbourhood of known depth.
 * Depth is estimated from the depth and depth gradient of pixels
 * in the neighbourhood, weighted by color and distance from
 * the target pixel.
 *
 * The pixels are filled in a particular order, to first fill
 * areas of uniform color in the image, followed by potential
 * edges.
 *
 * The full algorithm in the original paper also calls for
 * a guided filter after the first algorithm. The guided filter
 * is implemented separately.
 *
 * For details please refer to the original paper (Guided Depth
 * Enhancement via a Fast Marching Method).
 * */
class GDFMM {
  public:
  /** \brief Creates the exponential lookup tables
   * for the algorithm.
   * */
  GDFMM(float sigmaDistance = 2.0f,
        float sigmaColor = 10.0f,
        float blurSigma = 1.0f,
        int windowSize = 11);
  /** \brief Inpaints missing depths 
   *
   * \param[out] output If `output` is a pointer, the `cv::Mat` it
   * points to will be overwritten with the result.
   * */
  cv::Mat InPaint(const cv::Mat &depthImage,
                const cv::Mat &rgbImageOriginal,
                cv::Mat *output = nullptr);

  /** \brief An experimental alternative method to predict the depth
   * of unknown pixels by least-squares regression.
   *
   * Instead of predicting unknown depths with gradients, predict
   * the depth by running a regression on the neighbouring pixel's
   * RGB values.
   *
   *
   * @param[in] depthImage The depth image with missing values.
   * @param[in] rgbImageOriginal The reference image, either 8-bit 
   * three-channel, or one-bit. Computation is performed in 
   * single-precision floats, so don't bother with higher precisions.
   * @param[in] epsilon	The ridge-regression epsilon
   * @param[in] constant The constant to homogenize the coordinates.
   * @param[in] truncation In every window with a depth range of
   * 		\f$\Delta f = f_\mathrm{max} - f_\mathrm{min}\f$, constrain depths
   * 		to the range \f$[f_\mathrm{min} - t\Delta f,
   * 		f_\mathrm{max} + t\Delta f] \f$
   * */
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
  unsigned int windowSize_, blurSigma_;
};
/** \brief Guided filter. Apply this to the image for the full algorithm.
 *
 * */
cv::Mat GuidedFilter(const cv::Mat &object,
                     const cv::Mat &reference,
                     cv::Mat *output,
                     int windowSize,
                     float epsilon);

/** \brief Internally-used class
 **/
struct Point {
  int x, y;
};

}
