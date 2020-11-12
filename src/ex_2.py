# การคำนวณคุณสมบัติเชิงโครงสร้าง (structural properties) ของโมเลกุลและการเตรียม dataset

import numpy as np
from scipy.spatial import distance

# http://www.quantum-machine.org/gdml/data/npz/benzene_dft.npz
data = np.load("dataset/benzene_dft.npz")
atom = data["z"]  # (12,)
coor = data["R"]  # (49863, 12, 3)
ener = data["E"]  # (49863,)

# 1. Bond distance
# 2. Bond angle
# 3. R, E, dist, angle ---> features.npz

# Ref: https://cccbdb.nist.gov/geom3x.asp?method=25&basis=19


def length(a, b):
    return distance.euclidean(a, b)


def angle(a, b, c):
    v1 = a - b
    v2 = c - b
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_vec = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_vec)  # in radian
    angle = angle * 180.0 / np.pi  # in degree
    return angle


num_bz = coor.shape[0]

# Distance
all_dist = np.zeros((num_bz, 6))
for i in range(num_bz):
    all_dist[i][0] = length(coor[i][0], coor[i][1])
    all_dist[i][1] = length(coor[i][1], coor[i][2])
    all_dist[i][2] = length(coor[i][2], coor[i][3])
    all_dist[i][3] = length(coor[i][3], coor[i][4])
    all_dist[i][4] = length(coor[i][4], coor[i][5])
    all_dist[i][5] = length(coor[i][5], coor[i][0])

print(all_dist.shape)
print(all_dist[:3])

### distance matrix ###
# bz1 = coor[0][:6]  # coordinates for first 6 carbons in benzene conformer
# dist = distance.cdist(bz1, bz1)
# print(dist)
# new = np.triu(dist)
# print(new)
# dist = dist[(new > 1) & (new < 2)]
# print(dist)
#######################

all_angle = np.zeros((num_bz, 6))
for i in range(num_bz):
    all_angle[i][0] = angle(coor[i][0], coor[i][1], coor[i][2])
    all_angle[i][1] = angle(coor[i][1], coor[i][2], coor[i][3])
    all_angle[i][2] = angle(coor[i][2], coor[i][3], coor[i][4])
    all_angle[i][3] = angle(coor[i][3], coor[i][4], coor[i][5])
    all_angle[i][4] = angle(coor[i][4], coor[i][5], coor[i][0])
    all_angle[i][5] = angle(coor[i][5], coor[i][0], coor[i][1])

print(all_angle.shape)
print(all_angle[-3:])

f = "dataset/benzene_features.npz"
np.savez_compressed(f, R=coor, dist=all_dist, angle=all_angle, E=ener)
