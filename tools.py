import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

r = 2
delta = 2
x, y = np.mgrid[-r:r:0.05, -r:r:0.05]
z = (1/2*math.pi*delta**2)*np.exp(-(x**2+y**2)/2*delta**2)
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
