from distutils.core import setup,Extension
import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([Extension("gdfmm",
                sources=['gdfmm.pyx',
                    ],
                include_dirs=[numpy.get_include(),
                      '../include',
                      ],
                libraries=['opencv_core',
                           'opencv_highgui',
                           'opencv_imgproc',
                           'opencv_calib3d',
                           'gdfmm'],
                library_dirs=['../build'],
                extra_compile_args=['-std=c++11'],
                language="c++",
                )]),
#   py_modules=['dsvh.py'],
    name='dsvh',
    data_files=[ ('lib', [
                '../build/libgdfmm.so'
                ]) ],
    version='0.0.1',
    )

