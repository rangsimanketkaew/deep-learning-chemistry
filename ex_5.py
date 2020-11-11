## ทำนายพลังงานการกลายเป็นอะตอม (atomization energy) ของโมเลกุลด้วย QM 7 dataset

import numpy as np
import tensorflow as tf 
# print(tf.__version__)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# 1. Load dataset
dataset = np.load('dataset/qm7.npz')
print(dataset.files)
input = dataset['input']
output = dataset['output']

print(f"shape of input: {input.shape}")
print(f"shape of output: {output.shape}")
# split dataset ==> training set & test set
X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)
print(f"shape of X_train: {X_train.shape}")
print(f"shape of X_test: {X_test.shape}")
print(f"shape of y_train: {y_train.shape}")
print(f"shape of y_test: {y_test.shape}")
print("")

# 2. Design and build model 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(621,)),            # input layer
    tf.keras.layers.Dense(256, activation='relu'),  # hidden layer
    tf.keras.layers.Dense(256, activation='relu'),  # hidden layer
    tf.keras.layers.Dense(256, activation='relu'),  # hidden layer
    tf.keras.layers.Dense(1)
])

# 3. Compile model
model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error)

model.summary()

# 4. Fit model
history = model.fit(
    X_train,
    y_train,
    epochs=100
)
model.save('model/demo_100epochs_relu_3hidden.h5')

# 5. Check loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Energy]')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)

# 6. Prediction
## x1, x2, x2, ... ==> f(x)
## f(x) ==> y
y_predict = model.predict(X_test).flatten()
print(y_predict.shape)
print(y_predict)

def plot_pred(y_test, y_predict):
    plt.axes(aspect='equal')
    plt.scatter(y_test, y_predict, s=1, c='b')
    plt.xlabel('True values [energy]')
    plt.ylabel('Prediction [energy]')
    lims = [-2300, -500]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims, c='r')
    plt.show()

plot_pred(y_test, y_predict)

# 7. Calculate MSE and RMSE
error = tf.keras.metrics.mean_squared_error(y_test, y_predict)
print(f"MSE  : {error}")
print(f"RMSE : {tf.sqrt(error)}")
