from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("smooth_cy", ["smooth_cy.pyx"],
                             include_dirs =  [numpy.get_include()],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],)]

)
