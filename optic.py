import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

alpha = 3
beta = 3
a = -4
b = 4
s = 1
om = 3
m = 20
h_xsi = (b - a) / m
h_x = (b - a) / m
x_h = np.arange(a, b, h_x)

def r(x, y):
    return np.sqrt(x * x + y * y)

def phi(x, y):
    return np.angle(x + 1j * y)

def fxy(x, y):
    return np.exp(-(r(x, y) * r(x, y)/s/s)) * np.exp(- (1j * phi(x, y) * om))


x = np.arange(-4, 4, h_x)
y = np.arange(-4, 4, h_x)
fxgrid, fygrid = np.meshgrid(x, y)
fzgrid = fxy(fxgrid, fygrid)

fig, axs = plt.subplots(2, 1)

axs[0].imshow(np.abs(fzgrid))
axs[0].set_title('Амплитуда исходной')
axs[1].imshow(np.angle(fzgrid))
axs[1].set_title('Фаза исходной')

plt.show()
