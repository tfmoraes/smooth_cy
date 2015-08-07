import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, sqrt, fabs, round
from cython.parallel import prange

DTYPE8 = np.uint8
ctypedef np.uint8_t DTYPE8_t

DTYPEF64 = np.float64
ctypedef np.float64_t DTYPEF64_t



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef inline DTYPE8_t GS(DTYPE8_t[:, :, :] I, int z, int y, int x) nogil:
    cdef int dz = I.shape[0]
    cdef int dy = I.shape[1]
    cdef int dx = I.shape[2]

    if 0 <= x < dx \
            and 0 <= y < dy \
            and 0 <= z < dz:
        return I[z, y, x]
    else:
        return 0



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void perim(DTYPE8_t[:, :, :] image,
                DTYPE8_t[:, :, :] out) nogil:

    cdef int dz = image.shape[0]
    cdef int dy = image.shape[1]
    cdef int dx = image.shape[2]

    cdef int z, y, x
    cdef int z_, y_, x_

    for z in prange(dz, nogil=True):
        for y in xrange(dy):
            for x in xrange(dx):
                for z_ in xrange(z-1, z+2):
                    for y_ in xrange(y-1, y+2):
                        for x_ in xrange(x-1, x+2):
                            if 0 <= x_ < dx \
                                    and 0 <= y_ < dy \
                                    and 0 <= z_ < dz \
                                    and image[z, y, x] != image[z_, y_, x_]:
                                out[z, y, x] = 1
                                break



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef DTYPEF64_t calculate_H(DTYPE8_t[:, :, :] I, int z, int y, int x) nogil:
    # double fx, fy, fz, fxx, fyy, fzz, fxy, fxz, fyz, H
    cdef DTYPEF64_t fx, fy, fz, fxx, fyy, fzz, fxy, fxz, fyz, H
    # int h, k, l
    cdef int h = 1
    cdef int k = 1
    cdef int l = 1

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

    if fx*fx + fy*fy + fz*fz > 0:
        H = ((fy*fy + fz*fz)*fxx + (fx*fx + fz*fz)*fyy \
                + (fx*fx + fy*fy)*fzz - 2*(fx*fy*fxy \
                + fx*fz*fxz + fy*fz*fyz)) \
                / (fx*fx + fy*fy + fz*fz)
    else:
        H = 0.0

    return H


# void replicate(Image_d source, Image_d dest){
	# int x, y, z;
	# for(z=0; z < source.dz; z++)
		# for(y=0; y < source.dy; y++)
			# for(x=0; x < source.dx; x++)
				# G(dest, z, y, x) = G(source, z, y, x);
# }

cdef void replicate(DTYPEF64_t[:, :, :] source, DTYPEF64_t[:, :, :] dest) nogil:
    cdef int dz = source.shape[0]
    cdef int dy = source.shape[1]
    cdef int dx = source.shape[2]
    cdef int x, y, z
    for z in prange(dz, nogil=True):
        for y in xrange(dy):
            for x in xrange(dx):
                dest[z, y, x] = source[z, y, x]

# Image_d smooth(Image image, int n){
	# int i, x, y, z, S;
	# double H, diff=0, dt=1/6.0, v, cn;
	# Image_d out, aux;

	# Image A1 = perim(image);
	# Image A2 = perim(A1);
	# Image A3 = perim(A2);
	# Image A4 = perim(A3);
	# Image Band = sum_bands(4, A1, A2, A3, A4);
	# free(A1.data);
	# free(A2.data);
	# free(A3.data);
	# free(A4.data);

	# out.data = (double *) malloc(image.dz*image.dy*image.dx*sizeof(double));
	# out.dz = image.dz;
	# out.dy = image.dy;
	# out.dx = image.dx;
	# aux.data = (double *) malloc(image.dz*image.dy*image.dx*sizeof(double));
	# aux.dz = image.dz;
	# aux.dy = image.dy;
	# aux.dx = image.dx;
    
    # out.sx = image.sx;
    # out.sy = image.sy;
    # out.sz = image.sz;

	# S = 0;
	# for(z=0; z < image.dz; z++){
		# for(y=0; y < out.dy; y++){
			# for(x=0; x < out.dx; x++){
				# if (G(image, z, y, x))
					# G(out, z, y, x) = 1;
				# else
					# G(out, z, y, x) = -1;
				# if (G(Band, z, y, x))
					# S += 1;
			# }
		# }
	# }

	# for(i=0; i < n; i++){
		# replicate(out, aux);
		# diff = 0.0;
		# for(z=0; z < out.dz; z++){
			# for(y=0; y < out.dy; y++){
				# for(x=0; x < out.dx; x++){
					# if (G(Band, z, y, x)){
						# H = calculate_H(aux, z, y, x);
						# v = G(aux, z, y, x) + dt*H;
						# if(G(image, z, y, x)){
							# G(out, z, y, x) = v > 0 ? v: 0;
						# } else {
							# G(out, z, y, x) = v < 0 ? v: 0;
						# }
						# diff += (G(out, z, y, x) - G(aux, z, y, x))*(G(out, z, y, x) - G(aux, z, y, x));
					# }
				# }
			# }
		# }
		# cn = sqrt((1.0/S) * diff);
		# printf("CN: %.28f - diff: %.28f\n", cn, diff);
		# if (cn <= E)
			# break;
	# }
	# return out;
# }


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def smooth(np.ndarray[DTYPE8_t, ndim=3] image,
           int n,
           np.ndarray[DTYPEF64_t,  ndim=3] out):

    cdef np.ndarray[DTYPE8_t, ndim=3] mask = np.zeros_like(image)
    cdef np.ndarray[DTYPEF64_t, ndim=3] aux = np.zeros_like(out)

    cdef int i, x, y, z, S;
    cdef DTYPEF64_t H, v, cn
    cdef DTYPEF64_t diff=0.0
    cdef DTYPEF64_t dt=1/6.0

    cdef DTYPEF64_t E = 0.00001

    for i in xrange(10):
        perim(image, mask)

    cdef int dz = image.shape[0]
    cdef int dy = image.shape[1]
    cdef int dx = image.shape[2]

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
        replicate(out, aux);
        diff = 0.0;

        for z in prange(dz, nogil=True):
            for y in xrange(dy):
                for x in xrange(dx):
                    if mask[z, y, x]:
                        H = calculate_H(aux, z, y, x);
                        v = aux[z, y, x] + dt*H;

                        if image[z, y, x]:
                            if v > 0:
                                out[z, y, x] = v
                            else:
                                out[z, y, x] = 0.0
                        else:
                            if v < 0:
                                out[z, y, x] = v
                            else:
                                out[z, y, x] = 0.0

                        diff += (out[z, y, x] - aux[z, y, x])*(out[z, y, x] - aux[z, y, x])

        cn = sqrt((1.0/S) * diff);
        print "CN: %.28f - diff: %.28f\n" % (cn, diff)

        if cn <= E:
            break;
