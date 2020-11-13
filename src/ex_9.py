# สอนการปรับแต่ง Loss function ใน TensorFlow

import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import numpy as np
import matplotlib.pyplot as plt

def GRMSE(y_true, y_pred):
    """Calculate Geometric root mean-squared error (GRMSE)
    """
    # convert numpy --> tensor
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    N = y_pred.shape[1]
    square_err = tf.math.square(tf.math.subtract(y_true, y_pred))
    mult_sum = tf.einsum('ij,ij->', square_err, square_err)
    return tf.math.pow(mult_sum, 1/(2*N))  # 2N^th root

## Training set
a = np.random.random((15, 1)) # a = [1, 3] --> b = [1, 9]
b = np.einsum('ij,ij->ij', a, a)
print(a)
print(b)

## Standalone usage test
print("-"*20)
grmse = GRMSE(a, b)
print(grmse)

## Neural network
# Define and build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss=GRMSE)

# Train model
history = model.fit(a, b, epochs=200)

def plot_loss(history):
    plt.plot(history.history['loss'], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Error [GRMSE]')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)

# Prediction
pred = model.predict([30])
print(f"\nPrediction: {pred}")
