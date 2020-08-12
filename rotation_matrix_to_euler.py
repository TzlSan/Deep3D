import sys
import math
from math import asin, atan2, cos
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0]);

    singular = (sy < 1e-6)

    if not singular: 
        x = math.atan2(R[2, 1] , R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z]) 

def close_enough(a, b, epsilon = sys.float_info.epsilon):
    return (epsilon > abs(a - b))

def rotation_matrix_to_euler2(R):
    PI = 3.14159265358979323846264
    if close_enough(R[0, 2], -1.0):
        x = 0
        y = PI / 2
        z = x + atan2(R[1, 0], R[2, 0])
        return np.array([x, y, z])

    if close_enough(R[0, 2], 1.0): 
        x = 0
        y = -PI / 2
        z = -x + atan2(-R[1, 0], -R[2, 0])
        return np.array([x, y, z])
    else: 
        x1 = -asin(R[0, 2]);
        x2 = PI - x1;

        y1 = atan2(R[1, 2] / cos(x1), R[2, 2] / cos(x1))
        y2 = atan2(R[1, 2] / cos(x2), R[2, 2] / cos(x2))

        z1 = atan2(R[0, 1] / cos(x1), R[0, 0] / cos(x1))
        z2 = atan2(R[0, 1] / cos(x2), R[0, 0] / cos(x2))

        if (abs(x1) + abs(y1) + abs(z1)) <= (abs(x2) + abs(y2) + abs(z2)):
            return np.array([x1, y1, z1])
        else:
            return np.array([x2, y2, z2])


if __name__=="__main__":

    #a = np.genfromtxt('004158.txt').astype(np.float32).reshape((3, 4))
    a = np.genfromtxt('004164.txt').astype(np.float32).reshape((3, 4))
    b = np.genfromtxt('004152.txt').astype(np.float32).reshape((3, 4))

    a = np.concatenate([a, np.array([[0, 0, 0, 1]])], axis=0)
    b = np.concatenate([b, np.array([[0, 0, 0, 1]])], axis=0)

    a = np.linalg.inv(a)
    b = np.linalg.inv(b)

    a_r = list(a[:3, :3])
    b_r = list(b[:3, :3])

    A_r = R.from_matrix(a_r)
    B_r = R.from_matrix(b_r)
    
    #a_euler = A_r.as_euler('zyx', degrees=True)
    #b_euler = B_r.as_euler('zyx', degrees=True)
    #print(a_euler)

    a_euler = rotation_matrix_to_euler(a)
    b_euler = rotation_matrix_to_euler(b)
    print(a_euler)

    #a_euler = rotation_matrix_to_euler2(a)
    #b_euler = rotation_matrix_to_euler2(b)
    #print(a_euler)

    r1, r2, r3 = 2 * a_euler - 2 * b_euler
    t1, t2, t3 = a[:3, -1:] - b[:3, -1:]
    e = np.array([r1, r2, r3, t1[0], t2[0], t3[0]])
    e_norm = np.linalg.norm(e)
    print(e_norm)
