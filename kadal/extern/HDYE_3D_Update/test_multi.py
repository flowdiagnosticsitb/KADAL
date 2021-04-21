import numpy as np
import kmac

dat="""-4.632504 -8.862274 -5.369237 
-1.748896 -9.845880 -3.447994 
-7.108236 -7.033704 -9.625512 
-6.209090 -7.838827 -6.780259 
-6.446806 -7.644520 -2.314544 
-5.243587 -8.514975 -1.138873 
-2.310623 -9.729389 -9.705481 
-9.573677 -2.888719 -0.198476 
-9.996871 -0.250155 -4.509609 
-1.022884 -9.947548 -4.678429 
"""
dat = np.fromstring(dat, sep=' ').reshape(-1, 3)
ref_point = np.array([-10, -10, -10])
mean_vector = np.array([[-2.917795, -9.564856, -3.172051],
                        [-2.9, -9.5, -3.2],
                        [-6.605903, -7.507466, -1.988600],])
std_dev = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],])
ehvi = np.zeros(3)
ehvi[:] = kmac.ehvi3d_sliceupdate_multi(dat, ref_point, mean_vector, std_dev)
print(ehvi)
ans = np.array([3.707616667, 4.563448328, 0])
assert np.isclose(ehvi, ans).all()
print('Assert passed.')

print('Running infinite loop - check memory use for no leaks')
while True:
    ehvi[:] = kmac.ehvi3d_sliceupdate_multi(dat, ref_point, mean_vector, std_dev)


