import numpy as np
import matplotlib.pyplot as plt
import itertools as itertools

def ascomplex(a):
    return np.array(a, dtype=np.complex)


def integral(n, step, xs_f, ys_f, xs_F):
    shape = (n, n, n, n)

    # 1ое измерение
    x_4d = np.broadcast_to(xs_f[:, np.newaxis, np.newaxis, np.newaxis], shape)
    # 2ое измерение
    y_4d = np.broadcast_to(xs_f[np.newaxis, :, np.newaxis, np.newaxis], shape)

    # третье измерение
    u_4d = np.broadcast_to(xs_F[np.newaxis, np.newaxis, :, np.newaxis], shape)
    # четвертое измерение
    v_4d = np.broadcast_to(xs_F[np.newaxis, np.newaxis, np.newaxis, :], shape)

    # экспонента
    A = np.exp((-2 * np.pi * 1j) * (x_4d * u_4d + y_4d * v_4d))

    # масштабируем по ys_f
    A = A * np.broadcast_to(ys_f[:, :, np.newaxis, np.newaxis], shape)

    int_weights = np.ones(n)
    int_weights[0] = 1 / 2
    int_weights[-1] = 1 / 2
    int_weights *= step

    # масштабируем d1 по весам
    A = A * np.broadcast_to(int_weights[:, np.newaxis, np.newaxis, np.newaxis], shape)
    # масштабируем d2 по весам
    A = A * np.broadcast_to(int_weights[np.newaxis, :, np.newaxis, np.newaxis], shape)

    ys_F = A
    ys_F = np.sum(ys_F, axis=0)
    ys_F = np.sum(ys_F, axis=0)

    return ys_F


def draw_2d(xs, ys):
    axes = [xs[0], xs[-1], xs[0], xs[-1]]

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(ys), extent=axes)
    plt.title('Амплитуда полученной')

    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(ys), extent=axes)
    plt.title('Фаза полученной')

n = 1 << 6
m = 1 << 8

a_f = 4

def r(x, y):
    return np.sqrt(x * x + y * y)

def phi(x, y):
    return np.angle(x + 1j * y)

def fxy(x, y):
    return np.exp(-(r(x, y) * r(x, y))) * np.exp(- (1j * phi(x, y) * 3))


f_2d = lambda a: np.exp(-(r(a[:, :, 0], a[:, :, 1]) * r(a[:, :, 0], a[:, :, 1])/1)) * np.exp(- (1j * phi(a[:, :, 0], a[:, :, 1]) * 6))
F_2d = lambda a: np.exp(-2 * np.pi * 1j * (a[:, :, 0] * a[:, :, 1] + a[:, :, 2] * a[:, :, 3]))

# prep
a_F = n ** 2 / (4 * a_f * m)

step_f = 2 * a_f / n

xs_f = np.linspace(-a_f, a_f, n)
xs_f_shifted = xs_f - step_f / 2
xs_F = np.linspace(-a_F, a_F, n)

Xs_f = np.reshape(list(itertools.product(xs_f, xs_f)), (n, n, 2))
Xs_f_shifted = np.reshape(list(itertools.product(xs_f_shifted, xs_f_shifted)), (n, n, 2))
Xs_F = np.reshape(list(itertools.product(xs_F, xs_F)), (n, n, 2))

ys_f = ascomplex(f_2d(Xs_f))
# integral
ys_F_integral = ascomplex(integral(n, step_f, xs_f, ys_f, xs_F))

draw_2d(xs_f, ys_F_integral)
plt.show()