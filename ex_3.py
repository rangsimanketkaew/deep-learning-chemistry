# เตรียม dataset สำหรับการ train model
import numpy as np
from scipy.io import loadmat

m = loadmat('dataset/qm7.mat')
# print(type(m))

# Extract features
Z = m['Z']
R = m['R']
X = m['X']
T = m['T'][0]
print(Z.shape)
print(R.shape)
print(X.shape) # (7165x23x23) ==> (7165x529)
print(T.shape) # (1, 7165)

print("Reshape")
R = R.reshape([7165, -1])
X = X.reshape([7165, -1])
print(R.shape)
print(X.shape)

print("Concatenate array") 
All = np.concatenate((Z, R, X), axis=1) # column-wise
print(All.shape)

## การบันทึก Numpy Array
np.savez_compressed('dataset/qm7.npz', input=All)