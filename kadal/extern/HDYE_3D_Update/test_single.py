import numpy as np
import kmac

dat="""-0.981894 -9.951677 -4.021090 
-1.675298 -9.858670 -3.970364 
-6.989333 -7.151868 -6.273483 
-6.524929 -7.577948 -6.841600 
-9.967894 -0.800677 -4.963238 
-7.503389 -6.610533 -8.704238 
-8.988774 -4.382002 -4.607492 
-9.909816 -1.339981 -6.117480 
-8.487371 -5.288151 -6.438087 
-4.616053 -8.870854 -1.834406 
"""
dat = np.fromstring(dat, sep=' ').reshape(-1, 3)
ref_point = np.array([-10, -10, -10])
mean_vector = np.array([-7.255624, -6.881563, -4.043903]).reshape(1, -1)
std_dev = np.array([0, 0, 0]).reshape(1, -1)
ehvi = kmac.ehvi3d_sliceupdate_multi(dat, ref_point, mean_vector, std_dev)
print(ehvi)
assert np.isclose(ehvi, 9.739142427674365)
print('Assert passed.')

print('Running infinite loop - check memory use for no leaks')
while True:
    ehvi = kmac.ehvi3d_sliceupdate_multi(dat, ref_point, mean_vector, std_dev)

