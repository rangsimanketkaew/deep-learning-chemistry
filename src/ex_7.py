## ทำนายพลังงานการกลายเป็นอะตอม (atomization energy) ของโมเลกุลด้วย QM 7 dataset

import numpy as np
import tensorflow as tf 
# print(tf.__version__)
import matplotlib.pyplot as plt 

# 1. Load dataset
R = np.load('dataset/qm7_R.npz')['R']
Z = np.load('dataset/qm7_Z.npz')['Z']
X = np.load('dataset/qm7_X.npz')['X']
T = np.load('dataset/qm7_T.npz')['T']
print(R.shape)
print(Z.shape)
print(X.shape)
print(T.shape)

# 2. Design amd build model 
## Multi-input model using Functional API

input_1 = tf.keras.Input(shape=(69,))
input_2 = tf.keras.Input(shape=(23,))
input_3 = tf.keras.Input(shape=(529,))

all_input = tf.keras.layers.Concatenate()([input_1, input_2, input_3]) # merge model
hidden = tf.keras.layers.Dense(256, activation='relu')(all_input)
output = tf.keras.layers.Dense(1)(hidden)

model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=output, name='MultiInput_Model')

# 3. Compile model
model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error)

model.summary()
tf.keras.utils.plot_model(model, to_file='model/plot_multi-input_model.png', show_shapes=True)

# 4. Fit model
history = model.fit(
    [R, Z, X],
    T,
    epochs=100
)
model.save('model/demo_multi-input_100epochs_relu.h5')

# 5. Check loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Energy]')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)
