import matplotlib
matplotlib.use('gtk')
import matplotlib.pyplot as plt
import numpy as np
import gdfmm
import skimage.io
import pdb

rgb = skimage.io.imread('/home/daniel/nyu_label/rgb/r-1294439283.377657-2381571548.png')
dep = skimage.io.imread('/home/daniel/nyu_label/rawdepth/r-1294439283.377657-2381571548.png')
dep = np.asarray(dep, dtype=np.uint16)

pdb.set_trace()
X = gdfmm.InpaintDepth(dep, rgb)

plt.imshow(X)
plt.show()

