README
======
Implementation of "Guided depth enhancement via fast marching method" (Xiaojin Gong et al 2013)

The algorithm in the paper consists of two steps: Guided fast marching method, followed by
a guided filter.

The algorithms have been implemented separately as:

  namespace gdfmm {
    class GDFMM {
      cv::Mat InPaint(...);
    }
 
    cv::Mat GuidedFilter(...);
  }

Currently I have yet to work out the optimal mix of doubles/floats to trade off
accuracy and speed.

