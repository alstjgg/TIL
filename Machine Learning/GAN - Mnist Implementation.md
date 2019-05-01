# GAN implementation on tensorflow
- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [Tensorflow-Tutorials](https://github.com/golbin/TensorFlow-Tutorials/tree/master/09%20-%20GAN)
- [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/)
## Overview
- Based on mnist dataset
- Create images that are similar to images provided in the mnist dataset
- Structure of model

![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN/GAN_structure.png)

## Source Code
### Import libraries
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MINST_data/", one_hot=True)
```
### Options
```python
# Learning Options
total_epoch = 100
batch_size = 100
learning_rate = 0.0002

# Network Layer Options
n_hidden = 256      # Number of hidden layers
n_input = 28 * 28       # Size of input data
n_noise = 128       # Size of noise
```
- `total_epoch`, `batch_size`, `learning_rate` -> Hyper-parameters
### Network Model
```python
# # GAN is an unsupervised learning model, so Y is omitted
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])     # Noise input

# Generator
# # Generator first Weight -> size of noise x number of hidden layers
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
# # Bias -> initialized to 0 for each hidden layer
G_b1 = tf.Variable(tf.zeros([n_hidden]))
# # Last layer / Output
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# Discriminator
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
# # Scalar value indicating how 'true' the output is
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))
```
- Define trainable variables
### Setup
```python
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output

def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

G = generator(Z)        # Generate random image based on noise
D_gene = discriminator(G)       # Process generated image -> minimize
D_real = discriminator(X)       # Process true image -> maximize
```
### Loss function
```python
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))
```
### Optimization goal
```python
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]
train_D = tf.train.AdamOptimizer(learning_rate).\
    minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).\
    minimize(-loss_G, var_list=G_var_list)
```
### Session
```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0
```
### Run
```python
for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        # Train D & G
        _, loss_val_D = sess.run([train_D, loss_D],
                                 feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G],
                                 feed_dict={Z: noise})

    print('Epoch: ', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('/Users/argos/PycharmProjects/tensorflow/sample/{}'
                    '.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('Complete')
```
## Result
```
...
Epoch: 0099 D loss: -0.7828 G loss: -1.725
Complete
```
- epoch 000
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/000.png)
- epoch 009
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/009.png)
- epoch 019
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/019.png)
- epoch 029
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/029.png)
- epoch 039
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/039.png)
- epoch 049
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/049.png)
- epoch 059
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/059.png)
- epoch 069
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/069.png)
- epoch 079
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/079.png)
- epoch 089
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/089.png)
- epoch 099
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN%20samples/099.png)
