# สอนฟังก์ชันคณิตศาสตร์พื้นฐานและการดำเนินการ Tensor ใน TensorFlow

import tensorflow as tf 
import tensorflow.math as tfm

# tf.math.abs
# tf.abs

a = tf.constant([3, 3, 3])
b = tf.constant([1, 1 ,1])

sum = tf.add(a, b)
diff = tf.subtract(a, b)
prod = tf.multiply(a, b)
quot = tf.divide(a, b)

print(sum)
print(diff)
print(prod)
print(quot)