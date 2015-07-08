#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib
matplotlib.use('gtk')
import matplotlib.pyplot as plt
import os
import gdfmm

missing_mask = (cv2.imread('missing_mask.png', cv2.CV_LOAD_IMAGE_UNCHANGED) == 0)

for i in xrange(100):
    if os.path.isfile('images/rgb%d.png' % i) and \
       os.path.isfile('images/dep%d.png' % i) and \
       os.path.isfile('images/missing%d.png' % i):

          bgr = cv2.imread('images/rgb%d.png' % i, cv2.CV_LOAD_IMAGE_UNCHANGED)
          rgb = cv2.cvtColor(bgr, cv2.cv.CV_BGR2Lab)
          dep = cv2.imread('images/dep%d.png' % i, cv2.CV_LOAD_IMAGE_UNCHANGED)

          missing = dep.copy()
          missing[missing_mask] = 0

          inpainted = gdfmm.InpaintDepth(missing,
                                           rgb,
                                           sigma_distance =2,
                                           sigma_color = 3,
                                           blur_sigma=2,
                                           window_size = 11)


          # scale the depths to some visible range
          dep_scaled = (dep / 10000.0).reshape(dep.shape + (1,)).repeat(3, axis=2)
          inp_scaled = (inpainted/ 10000.0).reshape(dep.shape + (1,)).repeat(3, axis=2)
          mis_scaled = (missing  / 10000.0).reshape(dep.shape + (1,)).repeat(3, axis=2)
          rgb_scaled = rgb / 255.0

          dep_scaled = np.asarray(dep_scaled, dtype=np.float)
          inp_scaled = np.asarray(inp_scaled, dtype=np.float)
          rgb_scaled = np.asarray(rgb_scaled, dtype=np.float)
          mis_scaled = np.asarray(mis_scaled, dtype=np.float)

          side_by_side = np.concatenate(
                  (np.concatenate( (rgb_scaled, dep_scaled), axis=0 ),
                  np.concatenate( (mis_scaled, inp_scaled), axis=0 )), axis=1)
          plt.figure(figsize=(13,13))
          plt.imshow(side_by_side)
          plt.show()





