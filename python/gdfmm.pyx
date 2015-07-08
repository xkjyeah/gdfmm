cimport numpy as np
import numpy as np

cdef extern from "<cstdio>":
    void printf(const char *c, void *p)

cdef extern from "<cstdlib>":
    void *memcpy(void *dst, const void *src, unsigned int n)

cdef extern from "<opencv2/core/core.hpp>":
    cdef int CV_8UC3
    cdef int CV_16U

cdef extern from "<opencv2/core/core.hpp>" namespace "cv":
    cdef cppclass Mat:
        int rows
        int cols
        unsigned char *data
        Mat(int rows, int cols, int typ)
        Mat(int rows, int cols, int typ, void *data)
        Mat()
        void create(int rows, int cols, int typ)

cdef extern from "../include/gdfmm/gdfmm.h" namespace "gdfmm":
    cdef cppclass GDFMM:
        GDFMM(float sigma_dist, float sigma_color, float blur_sigma, int wsize);
        Mat InPaint(const Mat &depth, const Mat &rgb) except +
        Mat InPaint2(const Mat &depth, const Mat &rgb, float epsilon, float constant) except +

def InpaintDepth2(np.ndarray[np.uint16_t, ndim=2, mode="c"] dep,
                 np.ndarray[np.uint8_t, ndim=3, mode="c"] rgb,
                 float epsilon,
                 float constant,
                 float sigma_distance = 1.0,
                 float sigma_color = 10,
                 float blur_sigma = 1,
                 int window_size = 7):
    cdef Mat rgbM, depM, rvM
    rv = np.zeros((dep.shape[0], dep.shape[1]), dtype=np.uint16, order="C")

    rgbM = Mat(rgb.shape[0], rgb.shape[1], CV_8UC3, np.PyArray_DATA(rgb))
    depM = Mat(dep.shape[0], dep.shape[1], CV_16U, np.PyArray_DATA(dep))

    cdef void *p
    p = np.PyArray_DATA(dep)
    printf("%p\n", p)
#    print atoi(int(p))

    cdef GDFMM *gdfmm;
    gdfmm = new GDFMM(sigma_distance, sigma_color, blur_sigma, window_size);
    rvM = gdfmm.InPaint2(depM, rgbM, epsilon, constant)

    del gdfmm
    memcpy(np.PyArray_DATA(rv),
           rvM.data,
           2*dep.shape[0]*dep.shape[1])
    return rv

def InpaintDepth(np.ndarray[np.uint16_t, ndim=2, mode="c"] dep,
                 np.ndarray[np.uint8_t, ndim=3, mode="c"] rgb,
                 float sigma_distance = 1.0,
                 float sigma_color = 10,
                 float blur_sigma = 1,
                 int window_size = 7):
    cdef Mat rgbM, depM, rvM
    rv = np.zeros((dep.shape[0], dep.shape[1]), dtype=np.uint16, order="C")

    rgbM = Mat(rgb.shape[0], rgb.shape[1], CV_8UC3, np.PyArray_DATA(rgb))
    depM = Mat(dep.shape[0], dep.shape[1], CV_16U, np.PyArray_DATA(dep))

    cdef void *p
    p = np.PyArray_DATA(dep)
#    print atoi(int(p))

    cdef GDFMM *gdfmm;
    gdfmm = new GDFMM(sigma_distance, sigma_color, blur_sigma, window_size);
    rvM = gdfmm.InPaint(depM, rgbM)

    del gdfmm
    memcpy(np.PyArray_DATA(rv),
           rvM.data,
           2*dep.shape[0]*dep.shape[1])
    return rv
