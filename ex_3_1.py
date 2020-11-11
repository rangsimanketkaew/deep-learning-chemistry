## ทดสอบการเปิดไฟล์ npz ด้วย Numpy package
import numpy as np

n = np.load('dataset/qm7.npz')
print(n.files) # ['input']

my_input = n['input']
print(my_input.shape) # (7165, 621)