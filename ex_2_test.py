import numpy as np
from scipy.spatial import distance


data = np.load("dataset/benzene_dft.npz")
atom = data["z"]
coor = data["R"]
ener = data["E"]


def angle(a, b, c):
    v1 = a - b
    v2 = c - b
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_vec = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_vec)  # in radian
    angle = angle * 180.0 / np.pi  # in degree
    return angle


def length(a, b):
    return distance.euclidean(a, b)


all_dist = np.zeros((coor.shape[0], 6))
for i in range(coor.shape[0]):
    all_dist[i][0] = length(coor[i][0], coor[i][1])
    all_dist[i][1] = length(coor[i][1], coor[i][2])
    all_dist[i][2] = length(coor[i][2], coor[i][3])
    all_dist[i][3] = length(coor[i][3], coor[i][4])
    all_dist[i][4] = length(coor[i][4], coor[i][5])
    all_dist[i][5] = length(coor[i][5], coor[i][0])

print(all_dist.shape)

### distance matrix ###
bz1 = coor[0][:6]  # Only carbons
dist = distance.cdist(bz1, bz1)
print(dist)
new = np.triu(dist)
print(dist[(new > 1) & (new < 2)])

all_dist_mat = np.zeros((coor.shape[0], 6))
for i in range(coor.shape[0]):
    carbons = coor[i][:6]
    dist = distance.cdist(carbons, carbons)
    new = np.triu(dist)
    all_dist_mat[i] = dist[(new > 1) & (new < 2)]

#######################

all_dist.sort()
all_dist_mat.sort()
print(np.array_equal(all_dist, all_dist_mat))

all_angle = np.zeros((coor.shape[0], 6))
for i in range(coor.shape[0]):
    all_angle[i][0] = angle(coor[i][0], coor[i][1], coor[i][2])
    all_angle[i][1] = angle(coor[i][1], coor[i][2], coor[i][3])
    all_angle[i][2] = angle(coor[i][2], coor[i][3], coor[i][4])
    all_angle[i][3] = angle(coor[i][3], coor[i][4], coor[i][5])
    all_angle[i][4] = angle(coor[i][4], coor[i][5], coor[i][0])
    all_angle[i][5] = angle(coor[i][5], coor[i][0], coor[i][1])

print(all_angle[0])

f = "dataset/benzene_feature.npz"
np.savez_compressed(f, R=coor, dist=all_dist, angle=all_angle, E=ener)
