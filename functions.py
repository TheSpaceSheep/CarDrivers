from math import sqrt, sin, cos, pi
import numpy as np


def distance(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def rad(angle_in_degrees):
    return 2*np.pi * angle_in_degrees / 360


def deg(angle_in_radians):
    return 360 * angle_in_radians / 2*np.pi


def cartesian_to_polar(x, y, mode='deg'):
    z = x + 1j*y

    if mode == 'deg':
        return abs(z), 360 * np.arctan2(x, y) / (2*np.pi)
    else:
        return abs(z), np.arctan2(x, y)/2*np.pi


def polar_to_cartesian(r, theta, mode='deg'):
    if mode == 'deg':
        theta = 360 * theta / (2*np.pi)
    z = r * np.exp(1j*theta)
    return z.real, z.imag


def rotate(x, y, a, cx, cy):
    """applies the COUNTER-CLOCKWISE rotation of angle a and center cx, cy
    to the point x, y

              >  point returned
             /
            /a
  (cx, cy) /_____> (x, y)

    """
    a = - 2*pi*a/360

    center = np.array([cx, cy])

    M = np.array([[cos(a), -sin(a)],
                  [sin(a), cos(a)]])

    v = np.array([x-cx, y-cy])
    result = center + np.dot(M, v)
    return result[0], result[1]


def rectangle_vertices(ox=0, oy=0, width=10, length=20, angle=0):
    """
       length
    B __________A
     |         |
     |    O    |  width
     |_________|
    C           D

    angle : counter-clockwise (in degrees)

    returns the coordinates of the four vertices
    """

    Ax, Ay = ox + length / 2, oy + width / 2
    Bx, By = ox - length / 2, oy + width / 2
    Cx, Cy = ox - length / 2, oy - width / 2
    Dx, Dy = ox + length / 2, oy - width / 2

    Ax, Ay = rotate(Ax, Ay, angle, ox, oy)
    Bx, By = rotate(Bx, By, angle, ox, oy)
    Cx, Cy = rotate(Cx, Cy, angle, ox, oy)
    Dx, Dy = rotate(Dx, Dy, angle, ox, oy)

    return Ax, Ay, Bx, By, Cx, Cy, Dx, Dy


def det(a, b, c, d):
    return a*d - b*c


def is_right_of_line(x, y, ax, ay, bx, by):
    if det(x-ax, y-ay, bx-ax, by-ay) > 0:
        return True
    else:
        return False


def sec_to_hmsc(seconds):
    h = int(seconds // 3600)
    m = int(seconds // 60 - (seconds // 3600) * 60)
    s = int(seconds % 60)
    c = int((100 * seconds) % 100)
    return h, m, s, c
