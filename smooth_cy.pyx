import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, sqrt, fabs, round, fmin, fmax
from libc.string cimport memcpy
from cython.parallel import prange


ctypedef np.float32_t IMAGEF_t
IMAGEF = np.float32

ctypedef fused image_t:
    np.uint8_t
    np.int16_t
    IMAGEF_t


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef inline image_t GS(image_t[:, :, :] I, int z, int y, int x) nogil:
    cdef int dz = I.shape[0]
    cdef int dy = I.shape[1]
    cdef int dx = I.shape[2]

    return I[z%dz, y%dy, x%dx]



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void perim(np.uint8_t[:, :, :] image,
                np.uint8_t[:, :, :] out) nogil:

    cdef int dz = image.shape[0]
    cdef int dy = image.shape[1]
    cdef int dx = image.shape[2]

    cdef int z, y, x
    cdef int z_, y_, x_

    for z in prange(dz, nogil=True):
        for y in xrange(dy):
            for x in xrange(dx):
                for z_ in xrange(z-1, z+2, 2):
                    for y_ in xrange(y-1, y+2, 2):
                        for x_ in xrange(x-1, x+2, 2):
                            if GS(image, z, y, x) != GS(image, z_, y_, x_):
                                out[z, y, x] = 1
                                break



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef IMAGEF_t calculate_H(IMAGEF_t[:, :, :] I, int z, int y, int x) nogil:
    # double fx, fy, fz, fxx, fyy, fzz, fxy, fxz, fyz, H
    cdef IMAGEF_t fx, fy, fz, fxx, fyy, fzz, fxy, fxz, fyz, H
    # int h, k, l
    cdef int h = 1
    cdef int k = 1
    cdef int l = 1
    cdef IMAGEF_t divisor

    fx = (GS(I, z, y, x + h) - GS(I, z, y, x - h)) / (2.0*h)
    fy = (GS(I, z, y + k, x) - GS(I, z, y - k, x)) / (2.0*k)
    fz = (GS(I, z + l, y, x) - GS(I, z - l, y, x)) / (2.0*l)

    fxx = (GS(I, z, y, x + h) - 2*GS(I, z, y, x) + GS(I, z, y, x - h)) / (h*h)
    fyy = (GS(I, z, y + k, x) - 2*GS(I, z, y, x) + GS(I, z, y - k, x)) / (k*k)
    fzz = (GS(I, z + l, y, x) - 2*GS(I, z, y, x) + GS(I, z - l, y, x)) / (l*l)

    fxy = (GS(I, z, y + k, x + h) - GS(I, z, y - k, x + h) \
            - GS(I, z, y + k, x - h) + GS(I, z, y - k, x - h)) \
            / (4.0*h*k)
    fxz = (GS(I, z + l, y, x + h) - GS(I, z + l, y, x - h) \
            - GS(I, z - l, y, x + h) + GS(I, z - l, y, x - h)) \
            / (4.0*h*l)
    fyz = (GS(I, z + l, y + k, x) - GS(I, z + l, y - k, x) \
            - GS(I, z - l, y + k, x) + GS(I, z - l, y - k, x)) \
            / (4.0*k*l)

    divisor = fx*fx + fy*fy + fz*fz
    if divisor == 0:
        divisor = 0.000001

    H = ((fy*fy + fz*fz)*fxx + (fx*fx + fz*fz)*fyy + (fx*fx + fy*fy)*fzz - 2*(fx*fy*fxy  + fx*fz*fxz + fy*fz*fyz)) / (divisor)
    return H


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void replicate(IMAGEF_t[:, :, :] source, IMAGEF_t[:, :, :] dest) nogil:
    cdef int dz = source.shape[0]
    cdef int dy = source.shape[1]
    cdef int dx = source.shape[2]
    cdef int x, y, z
    for z in prange(dz, nogil=True):
        for y in xrange(dy):
            for x in xrange(dx):
                dest[z, y, x] = source[z, y, x]


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def smooth(np.ndarray[np.uint8_t, ndim=3] image,
           int n, int bsize,
           tuple spacing):

    cdef np.uint8_t[:, :, :] mask = np.zeros_like(image)
    cdef np.uint8_t[:, :, :] _mask = np.zeros_like(image)
    cdef IMAGEF_t[:, :, :] out = np.zeros_like(image, dtype=IMAGEF)
    cdef IMAGEF_t[:, :, :] aux = np.zeros_like(out)

    cdef int i, x, y, z, S;
    cdef IMAGEF_t H, v, cn
    cdef IMAGEF_t diff=0.0
    cdef IMAGEF_t dt=1/6.0

    cdef IMAGEF_t E = 0.002

    cdef int dz = image.shape[0]
    cdef int dy = image.shape[1]
    cdef int dx = image.shape[2]

    cdef IMAGEF_t sx = spacing[0]
    cdef IMAGEF_t sy = spacing[1]
    cdef IMAGEF_t sz = spacing[2]

    #  _mask[:] = image
    memcpy(&_mask[0, 0, 0], &image[0, 0, 0], dz*dy*dx)
    for i in xrange(bsize):
        perim(_mask, mask)
        #  _mask[:] = mask
        memcpy(&_mask[0, 0, 0], &mask[0, 0, 0], dz*dy*dx)
        print i

    #  out[:] = mask
    del _mask


    S = 0
    for z in prange(dz, nogil=True):
        for y in xrange(dy):
            for x in xrange(dx):
                if image[z, y, x]:
                    out[z, y, x] = 1.0
                else:
                    out[z, y, x] = -1.0

                if mask[z, y, x]:
                    S += 1

    for i in xrange(n):
        #  replicate(out, aux);
        memcpy(&aux[0, 0, 0], &out[0, 0, 0], dz*dy*dx*sizeof(IMAGEF_t))
        diff = 0.0;

        for z in prange(dz, nogil=True):
            for y in xrange(dy):
                for x in xrange(dx):
                    if mask[z, y, x]:
                        H = calculate_H(aux, z, y, x);
                        v = aux[z, y, x] + dt*H;

                        if image[z, y, x]:
                            out[z, y, x] = fmax(v, 0.0)
                        else:
                            out[z, y, x] = fmin(v, 0.0)

                        diff += (out[z, y, x] - aux[z, y, x])*(out[z, y, x] - aux[z, y, x])

        cn = sqrt((1.0/S) * diff)
        print "%d - CN: %.28f - diff: %.28f - S=%d\n" % (i, cn, diff, S)

        if cn <= E:
            break

    return out
