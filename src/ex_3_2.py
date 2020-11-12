## การแบ่งชุดข้อมูลสำหรับการ train และการ test model ของเรา

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

## Input
n = np.load('dataset/qm7.npz')
# print(n.files)
my_input = n['input']
# print(my_input.shape)

## Output
m = loadmat('dataset/qm7.mat')
T = m['T'][0]

## Split
X_train, X_test, y_train, y_test = train_test_split(my_input, T, test_size=0.2, random_state=42)
print(X_train.shape) # (5732, 621)
print(X_test.shape) # (1433, 621)
print(y_train.shape) # (5732,)
print(y_test.shape) # (1433,)